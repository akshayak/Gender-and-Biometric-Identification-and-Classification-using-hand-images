import os
import math
import numpy as np
import cv2
import pandas as pd
from skimage.feature import hog
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from scipy import spatial
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from scipy.stats import norm
import random
import argparse
from tqdm import tqdm
from math import *
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
# %matplotlib inline
from csv import reader
from math import sqrt
from math import exp
from math import pi
dataset_path='./dataset/'
from jinja2 import Template



#hog feature extration technique to extract features for given image
def extractHOGforImage(img):
    scale_percent=10
    height = int(img.shape[0] * scale_percent / 100)
    width = int(img.shape[1] * scale_percent / 100)
    dim = (width, height)
    # downsizing image
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    features,_  = hog(resized_img, orientations=9, 
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), 
                       transform_sqrt=True, 
                       visualize=True, feature_vector=True)
    
    return features




parser = argparse.ArgumentParser(description='Task 1.')
parser.add_argument("--train",
                   help='Enter the full path for the train folder',default = "./dataset/Hands_Test/Labelled/Set2/")
parser.add_argument("--test", help='Enter full path for the test folder', default = "./dataset/Hands_Test/Unlabelled/Set 2/")
parser.add_argument("--train_csv", default = "./dataset/Hands_Test/labelled_set2.csv")
parser.add_argument("--test_csv", default = "./dataset/Hands_Test/11k.csv")
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                    const=sum, default=max,
#                    help='sum the integers (default: find the max)')
args = parser.parse_args()


dataset_path='./dataset/'
df = pd.read_csv(dataset_path+"Hands_Test/labelled_set1.csv") 
#train read

# folder_path=dataset_path+'Hands_Test/Labelled/Set1/'
# df = pd.read_csv(dataset_path+"Hands_Test/labelled_set1.csv") 
df = pd.read_csv(args.train_csv)
folder_path = args.train
labels = []
images=os.listdir(folder_path)
hog_result=[]
image_names = []
for image in tqdm(images):
    labels.append(df.loc[df['imageName'] == image].aspectOfHand.values[0].split()[0])
    img_path=folder_path+image
    img = cv2.imread(img_path)
    # print(img_path)
    image_names.append(img_path)
    hog_result.append(extractHOGforImage(img))
# np.save(dataset_path+"results/hog_descriptors_train_set1.npy" , hog_result)
# np.save(dataset_path+"results/labels_train_set1.npy" , labels)
# np.save(dataset_path+"results/x_train1_imagepaths.npy", img_path)
x_train = hog_result
del hog_result
y_train = labels
del labels
train_path = image_names


df = pd.read_csv(args.test_csv)
folder_path = args.test
labels = []
images=os.listdir(folder_path)
hog_result=[]
image_names = []
for image in tqdm(images):
    labels.append(df.loc[df['imageName'] == image].aspectOfHand.values[0].split()[0])
    img_path=folder_path+image
    img = cv2.imread(img_path)
    # print(img_path)
    image_names.append(img_path)
    hog_result.append(extractHOGforImage(img))
# np.save(dataset_path+"results/hog_descriptors_train_set1.npy" , hog_result)
# np.save(dataset_path+"results/labels_train_set1.npy" , labels)
# np.save(dataset_path+"results/x_train1_imagepaths.npy", img_path)
x_test = hog_result
del hog_result
y_test = labels
del labels
test_path = image_names






# x_train1 = np.load(dataset_path+"results/cm_descriptors_train_set1.npy")
# x_train2 = np.load(dataset_path+"results/cm_descriptors_train_set2.npy")
# x_test1 = np.load(dataset_path+"results/cm_descriptors_test_set1.npy")
# x_test2 = np.load(dataset_path+"results/cm_descriptors_test_set2.npy")
# y_train1 = np.load(dataset_path+"results/labels_train_set1.npy")
# y_train2 = np.load(dataset_path+"results/labels_train_set2.npy")
# y_test1 = np.load(dataset_path+"results/labels_test_set1.npy")
# y_test2 = np.load(dataset_path+"results/labels_test_set2.npy")
# x_train1 = np.reshape(x_train1,(100,1728))
# x_train2 = np.reshape(x_train2,(100,1728))
# x_test1 = np.reshape(x_test1,(100,1728))
# x_test2 = np.reshape(x_test2,(100,1728))

pca = PCA(n_components=30)
# pca2 = PCA(n_components=30)
x_train = pca.fit_transform(x_train)
# x_train2 = pca2.fit_transform(x_train2)
x_test = pca.transform(x_test)
# x_test1_using1 = pca1.transform(x_test1)
# x_test1_using2 = pca2.transform(x_test1)
# x_test2_using1 = pca1.transform(x_test2)
# x_test2_using2 = pca2.transform(x_test2)

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
		print('[%s] => %d' % (value, i))
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
 
# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated
 
# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
 
# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries
 
# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries
 
# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent
 
# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            # print(i,mean,stdev)
            if stdev == 0:
                stdev = 1
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities
 
# Predict the class for a given row
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label


def result_visualization(filename,imagespath,acc,y_test,y_pred):
    # imagespath=[]
    # #path='C:/Users/svasud13.ASURITE/Downloads/phase3_sample_data/phase3_sample_data/Unlabelled/Set 1'
    # for imageid in images:
    #     imagespath.append(path+imageid)
    # print(imagespath)
    i=0
    title1 = "Task 1 accuracy = "+str(acc)
    html_code="<style>\nimg{\n border-style:solid;width: 160px;height:160px;display:block;\n}\np{width:160px;display:inline-block;text-overflow:ellipsis;white-space:nowrap;overflow:hidden}\n</style><title>"+title1+"</title>\n"
    html_code+="<body>\n<br>\n<h1 align=center>"+title1+"</h1>"
    while i<len(imagespath):
        # print(imagespath[i])
        if y_test[i] == 1:
            true = "Dorsal"
        else:
            true = "Palmar"
        if y_pred[i] == 1:
            test = "Dorsal"
        else:
            test = "Palmar"
        html_code+="<div style='display:inline-block;margin:10px 5px 0 5px;'><img src='"+imagespath[i]+"'/>\n"
        html_code+="<p >"+imagespath[i].split("/")[-1]+"</p><br><p>"+"True label :"+true+"</p><br><p>"+"Predicted label :"+test+"</p></div>\n"
        html_code+="</div>\n</body>"
        i+=1
    file = open(filename+'.html', 'w+')
    t=Template(html_code)
    file.write(t.render(title="task1"))
    file.close()


def classify(x_train,y_train,x_test,y_test):
    y_train_new = []
    for i in y_train:
        if i == "dorsal":
            y_train_new.append(1)
        else:
            y_train_new.append(0)
    dataset = []
    for (i,j) in zip(x_train,y_train_new):
        i = list(i)
        i.append(j)
        dataset.append(i)
    model = summarize_by_class(dataset)

    
    y_test1 = []
    for i in y_test:
        if i == "dorsal":
            y_test1.append(1)
        else:
            y_test1.append(0)
    y_pred = []
    for i in x_test:
        y_pred.append(predict(model,i))

    print(accuracy_score(y_test1,y_pred))
    result_visualization("temp",test_path,accuracy_score(y_test1,y_pred),y_test1,y_pred)
# classify(x_train1,y_train1,x_test1_using1,y_test1)
# classify(x_train1,y_train1,x_test2_using1,y_test2)
# classify(x_train2,y_train2,x_test1_using2,y_test1)
# classify(x_train2,y_train2,x_test2_using2,y_test2)

classify(x_train,y_train,x_test,y_test)