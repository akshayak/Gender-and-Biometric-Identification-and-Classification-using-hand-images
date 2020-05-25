import os
import math
import numpy as np
import cv2
import scipy.stats
from pymongo import MongoClient
import pandas as pd
from sklearn.decomposition import PCA
from skimage.feature import hog
import array as arr 
import csv
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler




# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))
# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']


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
    return np.asarray(outputVector).flatten()





def SVD(data_feature_matrix,k):
    U, Sigma, VT = randomized_svd(np.array(data_feature_matrix),n_components=k,n_iter='auto',random_state=None)
    return U,Sigma,VT 

def readFolderImagesAndStoreColorMoments(folderName,meta):
    documents = []
    metaAspect=[]
    dir = os.listdir(folderName)

    for fileName in dir:
        if fileName.endswith('jpg') or fileName.endswith('jpeg') or fileName.endswith('png'):
            col_output = calculateColorMomentsForImage(folderName + "/" + fileName)
            file = open(meta, 'r')
            for row in csv.reader(file):
                if(row[8]==fileName):
                    if row[7][0]=='d':
                        metaAspect.append(0)
                    else:
                        metaAspect.append(1)
            documents.append(col_output)
    return documents,metaAspect

def CalculateColorMoments(folderName,docs):
    documents = []
    fileNames=[]
    labels=[]
    dir = os.listdir(folderName)
    for fileName in dir:
        if fileName.endswith('jpg') or fileName.endswith('jpeg') or fileName.endswith('png'):
            fileNames.append(fileName)
            col_output = calculateColorMomentsForImage(folderName + "/" + fileName)
            documents.append(col_output)
    docs = StandardScaler().fit_transform(np.asarray(docs))
    U,Sigma,VT=SVD(docs,30)
    documents = StandardScaler().fit_transform(np.asarray(documents))
    test=np.matmul((np.matmul(np.array(documents),np.transpose(np.array(VT)))),(np.linalg.inv(np.array(np.diag(Sigma)))))
    return U,test,fileNames



folderName=input("Enter the Folder of the training data set")
metaFileName=input("Enter the CSV file of the meta data of the training set")
testFolder=input("Enter the Folder of the test Data set")
documents,metaAspects=readFolderImagesAndStoreColorMoments(folderName,metaFileName)
U,test,fileNames=CalculateColorMoments(testFolder,documents)
treeDoc=[]
testDoc=[]
for i in range(len(metaAspects)):
    temp=[]
    for j in range(len(U[i])):
        temp.append(U[i][j])
    temp.append(metaAspects[i])
    treeDoc.append(temp)
for i in range(len(test)):
    temp=[]
    for j in range(len(test[i])):
        temp.append(test[i][j])
    temp.append(0)
    testDoc.append(temp)
tree = build_tree(treeDoc, 30, 46 )
stump = tree




for i in range(len(testDoc)):
    prediction = predict(stump, testDoc[i])
    f=""

    if prediction ==0:
        f="dorsal"
    else:
        f="palmar"
    print(fileNames[i],"    ",f)










