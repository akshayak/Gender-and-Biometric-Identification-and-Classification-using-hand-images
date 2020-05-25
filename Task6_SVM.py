from Task4_SVM import Support_Vector_Machine
from Task4_SVM import calculateColorMomentsForImage
import numpy as np
from sklearn.preprocessing import StandardScaler
from jinja2 import Template


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


def train_relevant_feedback(trainset_images, train_labels, test_set_images):
    # Extract color moments for the training images
    print(trainset_images)
    print(train_labels)
    train_feature_matrix = []
    test_feature_matrix = []
    for image in trainset_images:
        cm_vector = calculateColorMomentsForImage("C:/Users/aksha/Documents/Fall2019/Multimedia and Web databases(CSE515)/Project/Phase 3/Hands/" + image)
        train_feature_matrix.append(np.asarray(cm_vector).flatten())

    for image in test_set_images:
        cm_vector = calculateColorMomentsForImage("C:/Users/aksha/Documents/Fall2019/Multimedia and Web databases(CSE515)/Project/Phase 3/Hands/" + image)
        test_feature_matrix.append(np.asarray(cm_vector).flatten())

    print(train_labels)
    normalised_feature_vector = StandardScaler().fit_transform(np.asarray(train_feature_matrix))
    svm_model = Support_Vector_Machine()
    svm_model.fit(normalised_feature_vector, train_labels, pos_label="R", min_sv_alpha=0.0001)
    normalised_test_vector = StandardScaler().fit_transform(np.asarray(test_feature_matrix))
    prediction = svm_model.predict(np.asarray(normalised_test_vector))
    conf_scores = svm_model.get_confidence_score()

    relevant_list = []
    irrelevant_list = []
    for i in range(len(test_set_images)):
        if prediction[i] == 1:
            relevant_list.append({'image_id': test_set_images[i], 'prediction': prediction[i], 'confidence_score': conf_scores[i]})
        else:
            irrelevant_list.append({'image_id': test_set_images[i], 'prediction': prediction[i], 'confidence_score': conf_scores[i]})

    sorted_rel_list = sorted(relevant_list, key=lambda x: x['confidence_score'])
    sorted_irrel_list = sorted(irrelevant_list, key=lambda x: abs(x['confidence_score']))

    #res = sorted(img_scores, key=lambda i: 0 if i['confidence_score'] == 0 else -1 / i['confidence_score'])
    #print(res)

    sorted_rel_list.extend(sorted_irrel_list)
    print(sorted_rel_list)
    visual_img_ids = []
    for i in range(len(sorted_rel_list)):
        visual_img_ids.append(sorted_rel_list[i]['image_id'])
    print(visual_img_ids)
    #path = "E:/Sriram/11k_Hands/Hands_Dataset/"
    path="C:/Users/aksha/Documents/Fall2019/Multimedia and Web databases(CSE515)/Project/Phase 3/Hands/"
    dataset_path='C:/Users/aksha/Documents/Fall2019/Multimedia and Web databases(CSE515)/Project/Phase 3'
    result_visualization(dataset_path+"/SVM_Feedback", visual_img_ids, path)
    return


def main():
    
    cont = 'y'
    while cont == 'y':
        arrs = np.load("C:/Users/aksha/Documents/Fall2019/Multimedia and Web databases(CSE515)/Project/Phase 3/task_5_result.npy", allow_pickle=True)
        arrs = arrs.item()

        key = []

        for k in arrs.keys():
            key.append(k)
        print("Choose whether an image is relevant or not")
        print("R-Relevant I-Irrelevant")
        training = []
        labels = []
        testing = []
        for k in key:
            label = input(k + " ")
            if label == "R":
                training.append(k)
                labels.append("R")
            elif label == "I":
                training.append(k)
                labels.append("I")
            testing.append(k)

        train_relevant_feedback(training, labels, testing)
        cont = input("Do you want to continue? y-Yes n-No")


if __name__ == '__main__':
    main()
