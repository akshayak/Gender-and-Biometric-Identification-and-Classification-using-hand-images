import os
import cv2
import pandas as pd
import numpy as np
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from scipy import stats
from jinja2 import Template

image_rowDict = dict()
row_imageDict = dict()
beta = 0.5
entropy_value = 1e-15

# Function to extract HOG feature descriptor for given set of images
def featureExtractionHOG(img):
    scale = 10
    h = int(img.shape[0] * scale / 100)
    w = int(img.shape[1] * scale / 100)
    dim = (w, h)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    features, hog_image = hog(resized_img, orientations=9,pixels_per_cell=(8, 8),cells_per_block=(2, 2),transform_sqrt=True,visualize=True, feature_vector=True)
    return features


#Function to get Image Image Similarity using cosine similarity
def similarity(folderpath):
  #folder_path="C:/Users/svasud13.ASURITE/Downloads/phase3_sample_data/phase3_sample_data/Unlabelled/Set 1/"
  images=os.listdir(folderpath)
  hog_result=[]

  for image in images:
    img_path=folderpath+image
    img = cv2.imread(img_path)
    hog_result.append(featureExtractionHOG(img))

  np.savetxt("HOG_Features.csv" , hog_result, delimiter=",")

  df=pd.read_csv("HOG_Features.csv")
  sim_matrix=cosine_similarity(df, df)
  np.savetxt("II_Similarity.csv" ,sim_matrix, delimiter=",")


#Function to create Image Image Similarity graph with k edges input by user
def graphCreation(k,filepath):
    sim_mat = np.loadtxt("II_Similarity.csv",delimiter=',')
    image_set1=pd.read_csv(filepath)
    image_id = image_set1["imageName"]
    dictimg=dict(enumerate(list(image_id)))
    #print(dictimg)
    #print("Creating Image Image Similarity Graph")

    index=[np.argpartition(sim_mat[i], -(k+1))[-(k+1):] for i in range(0,len(sim_mat))]
    
    graph_matrix=np.zeros(sim_mat.shape)
    graph_edges=[]
    graph_edges_weight=[]
    for i in range(0,len(index)):
        for j in range(0,len(index[0])):
            if(i == index[i][j]):
                continue
            graph_matrix[i][index[i][j]] = sim_mat[i][index[i][j]]
            graph_edges.append((dictimg.get(i),dictimg.get(index[i][j]),sim_mat[i][index[i][j]]))
            graph_edges_weight.append(sim_mat[i][index[i][j]])

    np.savetxt("graphadjacencymatrix_"+str(k)+".csv",graph_matrix,delimiter=",")

    with open('graph_edge_'+str(k)+'.txt', 'w') as fp:
        fp.write('\n'.join('%s %s %s' % x for x in graph_edges))


    with open('graph_edgeweight_'+str(k)+'.txt', 'w') as fp:
        fp.write('\n'.join('%s' % x for x in graph_edges_weight))
    return


#Loading dataset path for PPR
def load_task_data(filepath):
    #image_set1=pd.read_csv("C:/Users/svasud13.ASURITE/Downloads/phase3_sample_data/phase3_sample_data/Unlabelled/unlablled_set1.csv")
    image_set1=pd.read_csv(filepath)
    imageId= image_set1["imageName"]
    i = 0
    for image in imageId:
        image_rowDict[image] = i
        row_imageDict[i] = image
        i = i + 1


#Function to formulate m_matrix using graphadjacency matrix created for given dataset
def m_matrix(adjacency_matrix):
    m = []
    for image in adjacency_matrix:
        image_sum = np.sum(image)
        if image_sum != 0:
            m.append(image/np.sum(image_sum))
        else:
            m.append(image)
    return np.matrix(m)


#Function to compute vq_vector
def vq_vector(mat_shape, image_ids):
    vq = np.zeros(shape = mat_shape)
    seed_val = 1.0 / len(image_ids)
    for image_id in image_ids:
        id = image_rowDict[image_id]
        vq[int(id)] = seed_val
    return vq


#Function to compute uq_vector
def uq_vector(m_matrix,vq_vector,entropy_value_1):
    uq = np.asmatrix(vq_vector).transpose()
    seed = uq
    entropy_val = 10.00000000000000000
    i = 0
    while entropy_val > entropy_value_1 and i < 5000:
        uq2 = beta*np.matmul(m_matrix,uq) + (1-beta)*(seed)
        entropy_val = abs(scipy.stats.entropy(uq2, uq))
        uq = uq2
        i = i + 1
    return np.asarray(uq)

#Function to diaplay given and dominant images
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


#main
'''Steps in execution
    1.Get dataset path from user and compute Image Image Similarity
    2.Get dataset metadata path from user
    3.Get input(k)and create graph with k edges
    4.Get 3 image ID's from user and input(K)- the number of dominant images
    5.Perform Personalized Page Ranking
    6.Display K dominant images'''
def main():
    dataset_path='C:/Users/aksha/Documents/Fall2019/Multimedia and Web databases(CSE515)/Project/Phase 3'
    print("Enter dataset path to perform PPR : ")
    folderpath = input()
    similarity(folderpath)
    print("Enter filepath for image metadata : ")
    filepath = input()
    k = int(input('Enter k(outgoing edge count) :'))
    graphCreation(k,filepath)
    load_task_data(filepath)
    print("Image-image Similarity Graph Created")
    print('Enter 3 Image Id')
    i1 = input()
    i2 = input()
    i3 = input()
    k_value=str(k)
    dominant=int(input('Enter K dominant images: '))
    adjacency_matrix_pandas = pd.read_csv("graphadjacencymatrix_"+k_value+".csv", header=None)
    adjacency_matrix = adjacency_matrix_pandas.values
    print("Computing Personalised Page Rank")
    m_mat = m_matrix(adjacency_matrix)
    vq_vec = vq_vector(m_mat.shape[0], [i1, i2, i3])
    uq = uq_vector(m_mat,vq_vec,entropy_value)
    uq = np.squeeze(np.asarray(uq))
    image_row = np.argsort(uq)[::-1][:dominant]
    image_id = []
    print("Image Id           Dominant Score")
    for row in image_row:
        print(row_imageDict[int(row)]," ",uq[int(row)])
        image_id.append(str(row_imageDict[int(row)]))

    path=dataset_path+'/Hands/'
    result_visualization(dataset_path+"/Personalized Page Ranking - Given Images",[str(i1),str(i2),str(i3)],path)
    result_visualization(dataset_path+"/Personalized Page Ranking - Dominant Images", image_id,path)


if __name__ == '__main__':
    main()