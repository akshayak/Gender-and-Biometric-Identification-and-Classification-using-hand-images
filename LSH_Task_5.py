#!/usr/bin/env python
# coding: utf-8

# **Locality Sensitive Hashing**

# In[ ]:


import os
import math
import numpy as np
import cv2
from scipy.integrate import quad
import pandas as pd
from scipy import spatial
import matplotlib.pyplot as plt
from jinja2 import Template


# In[ ]:


dataset_path='C:/Users/aksha/Documents/Fall2019/Multimedia and Web databases(CSE515)/Project/Phase 3'


# In[ ]:


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


# In[ ]:


def gaussian(x):
    mu = 0
    sigma = 0.25
    k = 1 / (sigma * math.sqrt(2*math.pi))
    s = -1.0 / (2 * sigma * sigma)

    return k * math.exp(s * (x - mu)*(x - mu))


# In[ ]:


def GaussProb_calc(x, y):
    Denominator, derr = quad(gaussian, 0, 1)
    Numerator, nerr = quad(gaussian, x, y)
    return Numerator/Denominator


# In[ ]:


def GaussianBand(d):
    Band = np.full(d, -1.0, dtype="float64")
    for index in range(len(Band)):
        Band[index] = GaussProb_calc(index/d, (index+1)/d)
    np.savetxt(dataset_path+"/bandValues.txt", Band)
    return Band


# In[ ]:


def fetch_metadata(imageId):
    #Reading Images from the Metadata
    imageIds_List = pd.read_csv(dataset_path+"/metadata.csv") 
    image_names=imageIds_List['imageName'].values.tolist()
    indexQImgId = image_names.index(imageId)
    return indexQImgId,image_names


# In[ ]:


def plot_figures(figures, nrows = 1, ncols=1):
    img = np.zeros([100,100,3],dtype=np.uint8)
    img.fill(255) 
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    for j in range(nrows):
        
        for ind,title in enumerate(figures[j]):
            axeslist.ravel()[ind + j*ncols].imshow(figures[j][title], cmap=plt.gray())
            axeslist.ravel()[ind + j*ncols].set_title(title)
            axeslist.ravel()[ind + j*ncols].set_axis_off()
        
        if ncols > len(figures[j]):
            for i in range(len(figures[j]),max(imgcount)):
                axeslist.ravel()[i + j*ncols].imshow(img, cmap=plt.gray())
                axeslist.ravel()[i + j*ncols].set_title('')
                axeslist.ravel()[i + j*ncols].set_axis_off()
    plt.tight_layout()
    plt.show()


# In[ ]:


def visualize(LshImageIdList):
    figures = []

    figures.append({'im'+str(i): plt.imread(dataset_path+'Hands/'+LshImageIdList[i]) for i in range(5)})
    figures.append({'im'+str(i): plt.imread(dataset_path+'Hands/'+LshImageIdList[i]) for i in range(5,10)})
    figures.append({'im'+str(i): plt.imread(dataset_path+'Hands/'+LshImageIdList[i]) for i in range(10,15)})
    figures.append({'im'+str(i): plt.imread(dataset_path+'Hands/'+LshImageIdList[i]) for i in range(15,20)})

    # plot of the images in a figure, with 2 rows and 3 columns
    plot_figures(figures, 4, 5)


# In[ ]:


def construct_lsh(L,K,imageId):
    w = 4
    b = np.full(shape=(L, K), fill_value=-1)
    # read the vector inputs from too

    count = 0
    randomVectors_List = []
    # fetch the image - feature matrix
    imagetfidf = np.load(dataset_path+"/object_latent_svd.npy")
    imagetfidf=imagetfidf.T
    data_matrix=imagetfidf
    imagetfidf = np.divide(imagetfidf,np.sum(imagetfidf,axis=1,keepdims=True))

    CountOfObjects, CountOfDimentions = imagetfidf.shape

    # get the randon vectors
    Band = GaussianBand(CountOfDimentions)

    # create the Index structure
    hash_LSH = np.full(shape=(L, K, CountOfObjects),
                   fill_value=-1, dtype="float64")
    Q_hash = np.full(shape=(L, K), fill_value=-1, dtype="float64")
    
    #band_index = 0
    for i in range(L):
        for j in range(K):
            randomVector = np.asarray(
                Band[np.random.permutation(CountOfDimentions)])
            #np.savetxt("Output/randVect.txt", randomVector)
            randomVectors_List.append(randomVector)
            b[i][j] = np.random.randint(w+1)
            for k in range(CountOfObjects):
                hash_LSH[i][j][k] = float(
                    np.dot(randomVector.T, imagetfidf[k]) + float(b[i][j]))/float(w)
    
    indexQImgId,image_names = fetch_metadata(imageId)
    
    for i in range(L):
        for j in range(K):
            Q_hash[i][j] = hash_LSH[i][j][indexQImgId]

    # normalize that data to above 10^8 to place them in different bins
    for i in range(L):
        for j in range(K):
            power = 6
            hash_LSH[i][j] = np.floor(hash_LSH[i][j] * math.pow(10, power))
            Q_hash[i][j] = np.floor(Q_hash[i][j] * math.pow(10, power))

    # Match the queryImage with the Hash_LSH

    bucket = set()

    for i in range(L):
        for j in range(K):
            for k in range(CountOfObjects):
                if hash_LSH[i][j][k] == Q_hash[i][j]:
                    count=count+1
                    bucket.add(k)
          

    LshImageIdList = []
    for imageIndex in bucket:
        LshImageIdList.append(str(image_names[imageIndex]))
    
    return LshImageIdList,count,data_matrix,bucket


# In[ ]:


def getResult():
    print("Enter the number of Layer")
    L = int(input())
    print("Enter the number of hashes per layer")
    K = int(input())
    print("Enter the Image Id")
    imageId=input()
    print("Enter the value for T")
    t=int(input())
    LshImageIdList,total_image,data_matrix,bucket=construct_lsh(L,K,imageId)
    data_feature_matrix=data_matrix[list(bucket),:]
    indexQImgId,image_names = fetch_metadata(imageId)
    query_image=data_matrix[indexQImgId].reshape(1,-1)
    save_dict = {}
    for i in bucket:
        compare_mat=data_matrix[i,:].reshape(1,-1)
        currVal=1 - spatial.distance.cosine(query_image, compare_mat)
        save_dict[image_names[i]] = currVal
        
    count=0
    visualize_list=[]
    print("Top {} images:".format(t))
    result_dict={}
    for key, value in sorted(save_dict.items(), key=lambda item: item[1],reverse=True):
        if(count>t):
            break
        visualize_list.append(key)
        result_dict[key]=value
        print(key+"\t"+str(value))
        count=count+1
    print('\nUnique Image {}\t Image touched {}'.format(len(LshImageIdList),total_image))
    
    path=dataset_path+'/Hands/'
    result_visualization(dataset_path+"/Task_5_visualized",visualize_list,path)
    
    np.save(dataset_path+"/task_5_result.npy", result_dict)


# In[ ]:


getResult()

