from multiprocessing import Pool
import numpy as np
import pandas as pd
#from constants import task3_input_file,task6_part2_input_file_name
from task3 import  m_matrix,uq_vector,vq_vector,load_task_data,row_imageDict,result_visualization

label_pair = dict()
result = dict()
labelled_row = dict()
labelled_img = dict()
entropy_val = 1e-15

def read_image_label_pairs():
    content=pd.read_csv("C:/Users/svasud13.ASURITE/Downloads/phase3_sample_data/phase3_sample_data/labelled_set1.csv")
    #content=pd.read_csv("C:/Users/svasud13.ASURITE/Downloads/phase3_sample_data/phase3_sample_data/Unlabelled/unlablled_set1.csv")
    list1=content['imageName']
    list2=content['aspectOfHand']
    image_labels=list(zip(list1,list2))
    print(image_labels[0])
    for il in image_labels:
        if il[1] in label_pair.keys():
            val = label_pair[il[1]]
            val.append(il[0])
            label_pair[il[1]] = val
        else:
            label_pair[il[1]] = [il[0]]


def main():
    read_image_label_pairs()
    print("Enter filepath for image metadata : ")
    filepath = input()
    load_task_data(filepath)
    print("Given Data after Classification")
    for i in label_pair.keys():
        print(i," ",label_pair[i])
    print("Loading Unlabelled Data")
    adjacency_matrix_pandas = pd.read_csv("graphadjacencymatrix_5"+".csv", header=None)
    adjacency_matrix = adjacency_matrix_pandas.values
    print(adjacency_matrix)
    m_mat = m_matrix(adjacency_matrix)
    for key in label_pair.keys():
        print("Personalised Page Rank for ", key)
        #print(label_pair[key])
        #print(m_mat.shape[0])
        vq = vq_vector(m_mat.shape[0],label_pair[key])
        result[key] =uq_vector(m_matrix,vq,entropy_val)


    for key in labelled_img.keys():
        result_visualization("Task 4 - PPR classifier " + str(key), labelled_img[key],filepath)



if __name__ == '__main__':
    main()