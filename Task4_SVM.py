import numpy as np
import cv2
import scipy.stats
import os
import pymongo
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import hog
import pandas as pd
import pickle
import array as arr
from cvxopt import matrix, solvers
import csv

def getDBConnection():
    client = pymongo.MongoClient("mongodb://127.0.0.1:27017/?compressors=disabled&gssapiServiceName=mongodb")
    return client.MWDBPhase2


# Calculate HOG for a particular image
def extractHOGforImage(imageId):
    img = cv2.imread(imageId)
    scale_percent = 10
    height = int(img.shape[0] * scale_percent / 100)
    width = int(img.shape[1] * scale_percent / 100)
    dim = (width, height)
    # downsizing image
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    features, hog_image = hog(resized_img, orientations=9,
                              pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2),
                              transform_sqrt=True,
                              visualize=True, feature_vector=True)
    hist = arr.array('d', features)
    histlist = hist.tolist()
    return histlist


def readFolderImagesAndStoreHOG(folderName):
    # truncate_collection("img_hog_output"i)
    database = getDBConnection()
    output_collection = database['img_hog_output']
    documents = readFolderAndGetHOG(folderName)
    output_collection.insert_many(documents)


def readFolderAndGetHOG(folderName):
    dir = os.listdir(folderName)
    documents = []
    for fileName in dir:
        if fileName.endswith('jpg') or fileName.endswith('jpeg') or fileName.endswith('png'):
            hog_output = extractHOGforImage(folderName + "/" + fileName)
            documents.append({'image_id': fileName, 'output_vector': hog_output})
    return documents


# Given an ImageID, divide it into 100*100 blocks and find color moments for each block and return them
def calculateColorMomentsForImage(imageId):
    image = cv2.imread(imageId)
    if image is None:  # If the given Image is not present, exits
        print("Invalid image ID or Path does not exist")
        exit(-1)
    dimensions = image.shape
    # Convert RGB image into YUV
    yuvImage = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuvImgArr = np.asarray(yuvImage)

    # Divide the image into 100*100 blocks
    reshapedYUV = yuvImgArr.reshape(dimensions[1] // 100, 100, dimensions[0] // 100, 100, 3)
    blocks = np.swapaxes(reshapedYUV, 1, 2).reshape(-1, 100, 100, 3)

    outputVector = []

    for i in range(blocks.shape[0]):
        flatBlock = blocks[i].flatten()

        # Separate Blocks into separate Channels
        yChannelArr = flatBlock[0::3]
        uChannelArr = flatBlock[1::3]
        vChannelArr = flatBlock[2::3]

        # Color moments for Y Channel
        meanYChannel = np.mean(yChannelArr)
        stdDeviYChannel = np.std(yChannelArr)
        skewYChannel = scipy.stats.skew(yChannelArr)

        # Color moments for U Channel
        meanUChannel = np.mean(uChannelArr)
        stdDeviUChannel = np.std(uChannelArr)
        skewUChannel = scipy.stats.skew(uChannelArr)

        # Color moments for V Channel
        meanVChannel = np.mean(vChannelArr)
        stdDeviVChannel = np.std(vChannelArr)
        skewVChannel = scipy.stats.skew(vChannelArr)

        moment = [meanYChannel, stdDeviYChannel, skewYChannel, meanUChannel, stdDeviUChannel, skewUChannel,
                  meanVChannel, stdDeviVChannel, skewVChannel]

        # Output vector has n-blocks * 9 dimensions in the order as follows
        # n-blocks * [Mean-Y, SD-Y, Skew-Y, Mean-U, SD-U, Skew-U, Mean-V, SD-V, Skew-V]
        outputVector.append(moment)
    return outputVector

def truncate_collection(collectionName):
    database = getDBConnection()
    collection_name = database[collectionName]
    collection_name.drop()

# Task-2: Function to get a folder name.
# Calculate Color Moment feature vector for all images in the folder.
def readFolderImagesAndStoreColorMoments(folderName):
    truncate_collection("img_color_moment_output")
    database = getDBConnection()
    output_collection = database['img_color_moment_output']
    documents = readFolderImagesAndGetColorMoments(folderName)
    output_collection.insert_many(documents)


def readFolderImagesAndGetColorMoments(folderName):
    dir = os.listdir(folderName)
    documents = []
    for fileName in dir:
        if fileName.endswith('jpg') or fileName.endswith('jpeg') or fileName.endswith('png'):
            cm_output = calculateColorMomentsForImage(folderName + "/" + fileName)
            documents.append({'image_id': fileName, 'output_vector': cm_output})
    return documents


def pca_for_svm():
    db = getDBConnection()
    collection = db['img_color_moment_output']
    rs = collection.find()
    feature_vector = []
    image_ids = []
    for item in rs:
        feature_vector.append(np.asarray(item['output_vector']).flatten())
        image_ids.append(item['image_id'])

    # Normalise the feature vector before finding the co-variance
    normalised_feature_vector = StandardScaler().fit_transform(np.asarray(feature_vector))
    pca = PCA(n_components=int(99))
    reduced_matrix = pca.fit_transform(normalised_feature_vector)

    # Save PCA model to be used for Test dataset
    filename = 'pca_model_svm.sav'
    pickle.dump(pca, open(filename, 'wb'))
    return reduced_matrix


def get_training_labels(fpath):
    df = pd.read_csv(fpath)
    df = df[['aspectOfHand', 'imageName']]
    df = df.sort_values(by=['imageName'])
    trainset_metadata = []
    for i in range(len(df)):
        trainset_metadata.append(df.iloc[i]['aspectOfHand'].split(' ')[0])
    return trainset_metadata


class Support_Vector_Machine(object):
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.pos_label = None

    def rbf_kernel(self, X_data, Y_labels, sigma=.5):
        return np.exp(-(np.linalg.norm(X_data - Y_labels, ord=2) ** 2) / (2 * sigma ** 2))

    def fit(self, X_data, Y_labels, pos_label=1, C=0.5, min_sv_alpha=0.0001):
        # train with data
        self.data = np.asarray(X_data)
        self.Y_labels = np.asarray(Y_labels)
        self.labels = np.ndarray(shape=self.Y_labels.shape[0])
        self.C = C
        self.min_sv_alpha = min_sv_alpha

        if type(self.labels) == 'String' and pos_label == 1:
            print("Error in arguments. Please pass a value for pos_label.")
            return -1

        elif type(self.labels[0] == 'String' and type(pos_label) == 'String'):
            self.pos_label = pos_label
            for i in range(self.Y_labels.shape[0]):
                if self.Y_labels[i] == pos_label:
                    self.labels[i] = float(1)
                else:
                    self.labels[i] = float(-1)

        # print(self.rbf_kernel(self.data, Y_labels))

        self.K = np.zeros((self.data.shape[0], self.data.shape[0]))

        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[0]):
                self.K[i, j] = np.dot(self.data[i], self.data[j])

        Q = matrix(np.matmul(np.outer(self.labels, self.labels), self.K).astype('double'))
        p = matrix(np.ones(self.K.shape[0]) * -1)

        #G = matrix(np.diag(np.ones(self.data.shape[0]) * -1))
        #h = matrix(np.zeros(self.data.shape[0]))

        G_x = matrix(np.diag(np.ones(self.K.shape[0]) * -1))
        h_x = matrix(np.zeros(self.K.shape[0]))
        G_slack = matrix(np.diag(np.ones(self.K.shape[0])))
        h_slack = matrix(np.ones(self.K.shape[0]) * self.C)
        G = matrix(np.vstack((G_x, G_slack)))
        h = matrix(np.vstack((h_x, h_slack)))

        A = matrix(self.labels.astype('double'), (1, self.K.shape[0]))
        b = matrix(0.0)

        solution = solvers.qp(Q, p, G, h, A, b, kktsolver='ldl')
        print(np.asarray(solution['x']).shape)

        # the solver outputs the alphas (lagrange multipliers)
        self.alphas = np.asarray(solution['x'])[:, 0]
        print('Lagrange multipliers:')
        print(self.alphas)

        # save the support vectors (i.e. the ones with non-zero alphas) and their y values
        sv_index = np.where((self.min_sv_alpha < self.alphas) & (self.alphas < self.C))
        self.support_alphas = self.alphas[sv_index]
        self.support_vectors = self.data[sv_index]
        self.targets = self.labels[sv_index]

        print('\nNumber of support vectors:', self.support_vectors.shape[0], end='\n\n')

        self.b = 0
        for i, yi in enumerate(self.targets):
            m = yi
            for j, xj in enumerate(self.support_vectors):
                m -= self.alphas[j] * self.targets[j] * self.support_vectors[i]
            self.b += m
        self.b /= self.targets.shape[0]

        self.w = 0
        for i, yi in enumerate(self.targets):
            self.w += self.alphas[i] * self.targets[i] * self.support_vectors[i]

        self.w_norm = np.linalg.norm(np.asarray(self.w))

    def predict(self, X):
        self.scores = []
        scores = np.zeros(shape=(X.shape[0]))
        # print(self.support_alphas)
        # print(self.targets)
        # print(self.support_vectors)
        # print(X)
        for i, xi in enumerate(X):
            for j in range(self.support_vectors.shape[0]):
                scores[i] += self.support_alphas[j] * self.targets[j] * np.dot(self.support_vectors[j], xi)
            scores[i] += self.b[0]
            self.scores.append(scores[i])

        return np.sign(scores)

    def get_confidence_score(self):
        #print(self.scores)
        distances = []
        for i in range(len(self.scores)):
            distances.append(self.scores[i]/self.w_norm)
        return distances


def main():
    db = getDBConnection()
    print("TASK 4 - Support Vector Machine\n")
    labelled_image_path = input("\nEnter the Labelled Images Folder Path :  ")
    labelled_metadata_path = input("\nEnter the path of Metadata of Labelled dataset :  ")
    test_data_path = input("\nEnter the test data path:  ")

    readFolderImagesAndStoreColorMoments(labelled_image_path)

    # Train SVM
    train_feature_matrix = pca_for_svm()
    train_labels = get_training_labels(labelled_metadata_path)
    svm_model = Support_Vector_Machine()
    svm_model.fit(train_feature_matrix, train_labels, pos_label='dorsal')

    # Calculate Color Moments for the test set and calculate PCA using the saved model.
    cm = readFolderImagesAndGetColorMoments(test_data_path)
    test_set = []
    test_img = []
    for item in range(len(cm)):
        test_set.append(np.asarray(cm[item]['output_vector']).flatten())
        test_img.append(np.asarray(cm[item]['image_id']))

    normalised_test_set = StandardScaler().fit_transform(np.asarray(test_set))

    # load the model from disk
    loaded_pca_model = pickle.load(open('pca_model_svm.sav', 'rb'))
    reduced_test_set = loaded_pca_model.transform(normalised_test_set)

    prediction = svm_model.predict(reduced_test_set)

    test_img_labels = []
    img_metadata = db['hand_image_metadata']
    for i in range(len(test_img)):
        res = img_metadata.find({"imageName": str(test_img[i])})
        test_img_labels.append(res[0]['aspectOfHand'])

    result = []
    for i in range(len(prediction)):
        if prediction[i] == 1:
            result.append('dorsal')
        else:
            result.append('palmar')

    print(result)

    count = 0
    for i in range(len(result)):
        if (result[i] == test_img_labels[i]):
            count += 1
    print("SVM Classification Accuracy : " + str(count))

    with open('svm_result.csv', 'w', newline='\n') as file:
        writer = csv.writer(file)
        for i in range(len(test_img)):
            writer.writerow([test_img[i], result[i]])


if __name__ == '__main__':
    main()
