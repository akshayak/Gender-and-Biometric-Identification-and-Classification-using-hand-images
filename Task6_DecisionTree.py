import numpy as np
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
from jinja2 import Template




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
def predict(node, row,total):
    total=total+(row[node['index']]*row[node['index']])
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row,total)
        else:
            return node['left'],total	
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row,total)
        else:
            return node['right'],total


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

def readFolderImagesAndStoreColorMoments(training):
    documents = []
    dir = os.listdir("C:\\Users\\aksha\\Documents\\Fall2019\\Multimedia and Web databases(CSE515)\\Project\\Phase 3\\Hands")
    for fileNm in training:
        for fileName in dir:
            if fileNm==fileName:
                col_output = calculateColorMomentsForImage("C:\\Users\\aksha\\Documents\\Fall2019\\Multimedia and Web databases(CSE515)\\Project\\Phase 3\\Hands" + "\\" + fileName)
                documents.append(col_output)
                break
    return documents

def CalculateColorMoments(testing,docs):
    documents = []
    fileNames=[]
    dir = os.listdir("C:\\Users\\aksha\\Documents\\Fall2019\\Multimedia and Web databases(CSE515)\\Project\\Phase 3\\Hands")
    for fileNm in testing:
        for fileName in dir:
            if fileNm==fileName:
                fileNames.append(fileName)
                col_output = calculateColorMomentsForImage("C:\\Users\\aksha\\Documents\\Fall2019\\Multimedia and Web databases(CSE515)\\Project\\Phase 3\\Hands" + "\\" + fileName)
                documents.append(col_output)
                break
    docs = StandardScaler().fit_transform(np.asarray(docs))
    U,Sigma,VT=SVD(docs,30)
    documents = StandardScaler().fit_transform(np.asarray(documents))
    test=np.matmul((np.matmul(np.array(documents),np.transpose(np.array(VT)))),(np.linalg.inv(np.array(np.diag(Sigma)))))
    return U,test,fileNames
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

cont='y'
while cont=='y':
    arrs=np.load("C:\\Users\\aksha\\Documents\\Fall2019\\Multimedia and Web databases(CSE515)\\Project\\Phase 3\\task_5_result.npy",allow_pickle=True)
    arrs=arrs.item()

    # arrs={"Key":arr}
    # print(arrs)
    # arrs=arrs["Key"]
    # arrs={'Hand_0000674.jpg': 1.0, 'Hand_0000678.jpg': 0.7703858926817266, 'Hand_0000645.jpg': 0.5512571766029027, 'Hand_0000445.jpg': 0.5443224832549262, 'Hand_0001607.jpg': 0.5420997293042786, 'Hand_0001423.jpg': 0.5415050052856049, 'Hand_0000643.jpg': 0.5384953313370702, 'Hand_0001608.jpg': 0.5359350580696726, 'Hand_0001996.jpg': 0.5345098524371453, 'Hand_0010150.jpg': 0.5311251650903913, 'Hand_0010098.jpg': 0.5291815284123488, 'Hand_0004948.jpg': 0.5243987615657335, 'Hand_0000713.jpg': 0.520779663576515, 'Hand_0011313.jpg': 0.5177738633755903, 'Hand_0001995.jpg': 0.5094734586902335, 'Hand_0001372.jpg': 0.5092765037157306, 'Hand_0001550.jpg': 0.5044198381158354, 'Hand_0000666.jpg': 0.5007780959365533, 'Hand_0000462.jpg': 0.500078600711185, 'Hand_0006308.jpg': 0.4985986756314349, 'Hand_0000598.jpg': 0.49802352710123665}
    key=[]








    for k,v in arrs.items():
        key.append(k)
    print("Choose whether a image is relevant or not")
    print("R-Relevant I-Irrelevant")
    training=[]
    labels=[]
    testing=[]
    for k in key:
        label=input(k+" ")
        if(label=="R"):
            training.append(k)
            testing.append(k)
            labels.append(1)
        else:
            if(label=="I"):
                training.append(k)
                testing.append(k)
                labels.append(0)
            else:
                testing.append(k)


    documents=readFolderImagesAndStoreColorMoments(training)
    U,test,fileNames=CalculateColorMoments(testing,documents)
    treeDoc=[]
    testDoc=[]
    for i in range(len(labels)):
        temp=[]
        for j in range(len(U[i])):
            temp.append(U[i][j])
        temp.append(labels[i])
        treeDoc.append(temp)
    for i in range(len(test)):
        temp=[]
        for j in range(len(test[i])):
            temp.append(test[i][j])
        temp.append(0)
        testDoc.append(temp)

    tree = build_tree(treeDoc, 30,5)
    stump = tree
    print("########Results###################")
    relevance=[]
    irrelevance=[]
    for i in range(len(testDoc)):

        total=0
        prediction,tots = predict(stump, testDoc[i],total)
        f=""
        

        if prediction ==1:
            f="Relevant"
            relevance.append({"Id":fileNames[i],"Score":tots,"Label":f})
        else:
            f="Irrelevant"
            irrelevance.append({"Id":fileNames[i],"Score":tots,"Label":f})

    
        # print(fileNames[i],"    ",f,"Total",tots)
    imageIdList=[]
    objectweightPairs=sorted(relevance, key=lambda x : x['Score'], reverse=False)
            
    for o in objectweightPairs:    
        print(o["Id"])
        imageIdList.append(o["Id"])

    objectweightPairs=sorted(irrelevance, key=lambda x : x['Score'], reverse=True)
            
    for o in objectweightPairs:    
        print(o["Id"])
        imageIdList.append(o["Id"])
    
    dataset_path='C:/Users/aksha/Documents/Fall2019/Multimedia and Web databases(CSE515)/Project/Phase 3'
    path=dataset_path+'/Hands/'
    result_visualization(dataset_path+"/Task6_dtree",imageIdList,path)
    
    cont=input("Do you want to continue? y-Yes n-No")














