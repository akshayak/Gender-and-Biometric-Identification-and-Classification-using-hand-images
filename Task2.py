import numpy as np
import matplotlib.pyplot as plt, mpld3
from matplotlib import style
import pandas as pd 
import os
import math
import numpy as np
import cv2
import scipy.stats
import pymongo
from sklearn.decomposition import PCA
from skimage.feature import hog
import array as arr 
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from jinja2 import Template

dataset_path='C:/Users/aksha/Documents/Fall2019/Multimedia and Web databases(CSE515)/Project/Phase 3'

class K_Means:
    def __init__(self,k,tolerance,max_iterations):
        self.k=k
        self.tolerance=tolerance
        self.max_iterations=max_iterations

    #assign first k images as the initial centroid
    def fit(self,data,image_ids):
        self.centroids={}
        for i in range(self.k):
            self.centroids[i]=data[i]
        #print("Centroids:",self.centroids)
        counter=0


        for i in range(self.max_iterations):
            counter=counter+1
            print("*****Counter=",counter)
            self.classes={}
            self.classImages={}
            for i in range(self.k):
                self.classes[i]=[]
                self.classImages[i]=[]

            # print("Centroids",self.centroids)

            i=0
            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                #print("Distances: ",distances)
                #print("Min=",classification)
                self.classes[classification].append(features) #appending the image to the most similar cluster(class)
                self.classImages[classification].append(image_ids[i])
                i=i+1

            
            previous = dict(self.centroids)
            #print("Classes",self.classes)

            #recalculating ith centroid with the respective avg of the points in that class
            for classification in self.classes:
                self.centroids[classification]=np.average(self.classes[classification],axis=0)
            
            isOptimal=True

            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr_centroid=self.centroids[centroid]

                if np.sum((curr_centroid - original_centroid)/original_centroid * 100.0) > self.tolerance:
                    isOptimal=False
            
            if isOptimal:
                break

        #print(counter)

    def predict(self,data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        # print("Distances:",distances)
        classification = distances.index(min(distances))
        return min(distances),classification

        


def getLabel(filename):
    database = getDBConnection()
    collection = database['hand_image_metadata']
    rs = collection.find_one({'imageName': filename})
    return rs['aspectOfHand']

def getDBConnection():
    client = pymongo.MongoClient("mongodb://127.0.0.1:27017/?compressors=disabled&gssapiServiceName=mongodb")
    return client.MWDBPhase2

def truncate_collection(collectionName):
    database = getDBConnection()
    collection_name = database[collectionName]
    collection_name.drop()

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



# Calculate Color Moment feature vector for all images in the folder.
def readFolderImagesAndStoreColorMoments(folderName):
    truncate_collection("img_color_moment_output")
    dir = os.listdir(folderName)
    database = getDBConnection()
    output_collection = database['img_color_moment_output']
    documents = []
    for fileName in dir:
        if fileName.endswith('jpg') or fileName.endswith('jpeg') or fileName.endswith('png'):
            cm_output = calculateColorMomentsForImage(folderName + "/" + fileName)
            documents.append({'image_id': fileName, 'output_vector': cm_output})
    #print(documents[29]['output_vector'])
    output_collection.insert_many(documents)

def pca( fpath):
    readFolderImagesAndStoreColorMoments(fpath)
    client = pymongo.MongoClient("mongodb://127.0.0.1:27017/?compressors=disabled&gssapiServiceName=mongodb")
    db = client.MWDBPhase2
    collection = db['img_color_moment_output']
    rs = collection.find()
    feature_vector = []
    image_ids = []
    k_val=40
    for item in rs:
        feature_vector.append(np.asarray(item['output_vector']).flatten())
        image_ids.append(item['image_id'])
    print(np.asarray(feature_vector).shape)

    return feature_vector,image_ids


def result_visualization(filename,images,path):
    imagespath=[]
    #path='C:/Users/svasud13.ASURITE/Downloads/phase3_sample_data/phase3_sample_data/Unlabelled/Set 1'
    for imageid in images:
        imagespath.append(path+imageid)
    i=0
    html_code="<style>\nimg{\n border-style:solid;width: 160px;height:160px;display:block;\n}\np{width:160px;display:inline-block;text-overflow:ellipsis;white-space:nowrap;overflow:hidden}\n</style><title>{{title}}</title>\n"
    html_code+="<body>\n<br>\n<h1 align=center>{{title}}</h1>"
    while i<len(imagespath):
        html_code+="<div style='display:inline-block;margin:10px 5px 0 5px;'><img src='"+imagespath[i]+"'/>\n"
        html_code+="<p >"+imagespath[i].rsplit('/',1)[1].split('.',1)[0]+"</p></div>\n"
        html_code+="</div>\n</body>"
        i+=1
    file = open(filename+'.html', 'w+')
    t=Template(html_code)
    file.write(t.render(title=filename))
    file.close()

def main():
    fpath=input("Enter the path")
    test_folder=input("Enter test image folder")
    c_val=input("Enter the number of clusters")
    X,image_ids=pca(fpath)
    X_dorsal=[]
    X_palmar=[]
    image_ids_dorsal=[]
    image_ids_palmar=[]
    for i in range(len(image_ids)):
        label=getLabel(image_ids[i])
        if(label=="dorsal"):
            X_dorsal.append(X[i])
            image_ids_dorsal.append(image_ids[i])
        else:
            X_palmar.append(X[i])
            image_ids_palmar.append(image_ids[i])
    
    print("Dorsal size: ",len(X_dorsal))
    print("Palmar size: ",len(X_palmar))

    km_dorsal =K_Means(int(c_val),0.0001,500)
    km_dorsal.fit(X_dorsal,image_ids_dorsal)
    km_palmar =K_Means(int(c_val),0.0001,500)
    km_palmar.fit(X_palmar,image_ids_palmar)
    print("Dorsal Clusters")
    for i in range(km_dorsal.k):
        print("---CLASS",i)                                                                                                                                                                                                                                   
        #print(km_dorsal.classes[i])
        print(km_dorsal.classImages[i])

    print("Palmar Clusters")
    for i in range(km_palmar.k):
        print("---CLASS",i)                                                                                                                                                                                                                                   
        #print(km_palmar.classes[i])
        print(km_palmar.classImages[i])

    
    for count in range(len(km_dorsal.classImages)):
        path=dataset_path+'/Hands/'
        result_visualization(dataset_path+"/Dorsal Cluster"+str(count+1),km_dorsal.classImages[count],path)

    for count in range(len(km_palmar.classImages)):
        path=dataset_path+'/Hands/'
        result_visualization(dataset_path+"/Dorsal Cluster"+str(count+1),km_palmar.classImages[count],path)



    for count in range(len(km_dorsal.classImages)):
        print(count)
        w=10
        h=10
        fig=plt.figure(figsize=(10, 10))
        i=1
        ax = []
        for image in km_dorsal.classImages[count]:
            print(image)
            img = cv2.imread(fpath+"\\"+image)
            columns = 8
            rows = 7
            ax.append(fig.add_subplot(rows, columns, i))
            ax[-1].set_title(image,fontsize=6) 
            i=i+1
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            plt.imshow(img)
        count1=count+1
        fig.suptitle('Dorsal cluster : %i' %count1)
        plt.show()

    for count in range(len(km_palmar.classImages)):
        print(count)
        w=10
        h=10
        fig=plt.figure(figsize=(10, 10))
        i=1
        ax = []
        for image in km_palmar.classImages[count]:
            print(image)
            img = cv2.imread(fpath+"\\"+image)
            columns = 8
            rows = 7
            ax.append(fig.add_subplot(rows, columns, i))
            ax[-1].set_title(image,fontsize=6) 
            i=i+1
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            plt.imshow(img)
        count1=count+1
        fig.suptitle('Palmar cluster : %i' %count1)
        plt.show()

    #test_image="C:\Users\aksha\Documents\Fall2019\Multimedia and Web databases(CSE515)\Project\Phase 3\phase3_sample_data\phase3_sample_data\Unlabelled\Set 1\Hand_0006645.jpg"
    dir = os.listdir(test_folder)
    totalCount=0
    correctCount=0
    wrongCount=0

    for fileName in dir:
        ftr_test_image = calculateColorMomentsForImage(test_folder + "/" + fileName)
        
        print("Test image:",fileName)
        ftr_test_image=np.asarray(ftr_test_image).flatten()
        print("Test image feature: ",len(ftr_test_image))
        distance1,classification1=km_dorsal.predict(ftr_test_image)
        distance2,classification2=km_palmar.predict(ftr_test_image)

        #C:\Users\aksha\Documents\Fall2019\Multimedia and Web databases(CSE515)\Project\testset4
        print("Dorsal distance",distance1)
        print("Palmar distance",distance2)

        if(distance1<distance2):
            label="dorsal"
            print("Label : Dorsal")
            print("Classification cluster :",classification1)

        else:
            label="palmar"
            print("Label : Palmar")
            print("Classification cluster :",classification2)    
        
        totalCount+=1
        if(label==getLabel(fileName)):
            correctCount+=1
        else:
            wrongCount+=1
        
    print("Correct Count :",correctCount)
    print("Wrong count :",wrongCount)
    print("Total Count: ",totalCount)


    
            





if __name__ == "__main__":
	main()