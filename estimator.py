import cv2
import time
import pickle
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from skimage.feature import hog
import os


st.set_option('deprecation.showPyplotGlobalUse', False)

classLabel = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# print(os.getcwd())
path = ""

with open(os.path.join(path,"mlp.pkl"),"rb") as f:
    mlp = pickle.load(f)
with open(os.path.join(path,"SVC_HOG.pkl"),"rb") as f:
    svc_hog = pickle.load(f)
with open(os.path.join(path,"svc_pca.pkl"),"rb") as f:
    svc_pca = pickle.load(f)
with open(os.path.join(path,"mlp_pca.pkl"),"rb") as f:
    mlp_pca = pickle.load(f)
with open(os.path.join(path,"mlp_hog.pkl"),"rb") as f:
    mlp_hog = pickle.load(f)
with open(os.path.join(path,"svc_rbf.pkl"),"rb") as f:
    svc = pickle.load(f)
with open(os.path.join(path,"rfc_hog.pkl"),"rb") as f:
    rfc_hog = pickle.load(f)
with open(os.path.join(path,"xgb_hog.pkl"),"rb") as f:
    xgb_hog = pickle.load(f)
with open(os.path.join(path,"pca.pkl"),"rb") as f:
    pca = pickle.load(f)

st.markdown("<h1 style='color:red;'>CIFAR-10 IMAGE CLASSIFICATION</h1>",unsafe_allow_html=True)
st.write("")

ch = st.sidebar.radio("NAVIGATE",["PREDICTOR","PERFORMANCE","ABOUT DATASET","TEAM"])

if(ch == "PREDICTOR"):
    img = st.file_uploader("Upload Image",type=["jpg","png","jpeg"])
    if(img):
        img = Image.open(img)
        st.image(img,use_column_width="always")
        img_array = np.array(img)

        resize_img = cv2.resize(img_array,(32,32)) # for hog based models
        flat_img = resize_img.flatten()  

        modelsList = ["Support Vector Classifier", "Support Vector Classifier (PCA)","Support Vector Classifier (HOG)","Multi-Layer Perceptron","Multi-Layer Perceptron (PCA)","Multi-Layer Perceptron (HOG)","Random Forest Classifier (HOG)","XgBoost (HOG)"]
        select = st.selectbox("Select Estimator",modelsList,index=2)

        if(select == "Support Vector Classifier (HOG)"):
            fd , hog_im = hog(resize_img , orientations=9 , pixels_per_cell = (8,8),
                     cells_per_block = (2,2) , visualize = True ,  multichannel = True)
            pred = svc_hog.predict([fd])

        elif(select == "Support Vector Classifier"):
            pred = svc.predict([flat_img])

        elif(select == "Support Vector Classifier (PCA)"):
            flat_img = pca.transform([flat_img])
            pred = svc_pca.predict(flat_img)
        
        elif(select == "Multi-Layer Perceptron"):
            pred = mlp.predict([flat_img])

        elif(select == "Multi-Layer Perceptron (PCA)"):
            flat_img = pca.transform([flat_img])
            pred = mlp_pca.predict(flat_img)

        elif(select == "Multi-Layer Perceptron (HOG)"):
            fd , hog_im = hog(resize_img , orientations=9 , pixels_per_cell = (8,8),
                     cells_per_block = (2,2) , visualize = True ,  multichannel = True)
            pred = mlp_hog.predict([fd])

        elif(select == "Random Forest Classifier (HOG)"):
            fd , hog_im = hog(resize_img , orientations=9 , pixels_per_cell = (8,8),
                     cells_per_block = (2,2) , visualize = True ,  multichannel = True)
            pred = rfc_hog.predict([fd])

        elif(select == "XgBoost (HOG)"):
            fd , hog_im = hog(resize_img , orientations=9 , pixels_per_cell = (8,8),
                     cells_per_block = (2,2) , visualize = True ,  multichannel = True)
            pred = xgb_hog.predict([fd])

        if(st.button("Predict")):
            with st.spinner("Predicting ..."):
                time.sleep(2)
            st.success(classLabel[int(pred)].upper())

        

elif(ch == "PERFORMANCE"):

    st.write("<h3 style='text-align:center'>Accuracy Score of Base Models</h3>",unsafe_allow_html=True)

    with open(os.path.join(path,"models1.pkl"),"rb") as f:
        models = pickle.load(f)
    with open(os.path.join(path,"accuracy_base.pkl"),"rb") as f:
        accuracy1 = pickle.load(f)
    with open(os.path.join(path,"accuracy_pca.pkl"),"rb") as f:
        accuracy2 = pickle.load(f)
    with open(os.path.join(path,"accuracy_hog.pkl"),"rb") as f:
        accuracy3 = pickle.load(f)
    with open(os.path.join(path,"model2.pkl"),"rb") as f:
        models2 = pickle.load(f)

    plt.figure(figsize=(13,9))
    plt.bar(models,accuracy1,width=0.5,color = ["g","r","b","y","c","m","g","r","b"])
    for i in range(len(accuracy1)):
        plt.text(i,accuracy1[i],str(round(accuracy1[i]*100))+"%",ha="center",va="bottom") 
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Performance (BASE MODELS)")
    # plt.show()
    st.pyplot()

    st.write("<h3 style='text-align:center'>Accuracy Score of Base Models with PCA </h3>",unsafe_allow_html=True)
    st.write("<p style='text-align:center'>PCA(n_components = 500) which retains 90% Data</p>",unsafe_allow_html=True)

    plt.figure(figsize=(13,9))
    plt.bar(models,accuracy2,width=0.5,color = ["g","r","b","y","c","m","g","r","b"])
    for i in range(len(accuracy2)):
        plt.text(i,accuracy2[i],str(round(accuracy2[i]*100))+"%",ha="center",va="bottom") 
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Performance (PCA MODELS)")
    # plt.show()
    st.pyplot()

    st.write("<h3 style='text-align:center'>Accuracy Score of HOG Models</h3>",unsafe_allow_html=True)

    plt.figure(figsize=(13,9))
    plt.bar(models2,accuracy3,width=0.5,color = ["g","r","b","y","c"])
    for i in range(len(accuracy3)):
        plt.text(i,accuracy3[i],str(round(accuracy3[i]*100))+"%",ha="center",va="bottom") 
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Performance (HOG MODELS)")
    # plt.show()
    st.pyplot()

elif(ch == "ABOUT DATASET"):

    st.markdown("<h3>ABOUT CIFAR-10 DATASET</h3>",unsafe_allow_html=True)
    st.markdown("CIFAR stands for Canadian Institue For Advanced Research, they were collected by Alex Krizhevsky, Vinod Nair, Geoffrey Hinton.",unsafe_allow_html=True)
    st.markdown("CIFAR-10 Dataset is a collection of 60000 32x32 color images distributed equally in 10 classes, 6000 images per classes.",unsafe_allow_html=True)
    st.markdown("50000 training samples and 10000 testing samples. Dataset is divided in 5 training batches and 1 test batch, each with 10000 images.")
    st.markdown("Training samples contains 5000 images per class and testing samples contains 1000 images per class.")
    st.markdown("Commonly Used in Machine Learning and Computer Vision Algorithms.")
    st.markdown("Classes are mutually exclusive. There is no overlap between automobiles and trucks which look similar.")
    st.markdown("### Class Labels")

    df = pd.DataFrame({
        "Class Label" : classLabel
    })
    st.write(df)

    st.markdown("### Some Examples")

    # visualizing some samples of each class
    (X_train,y_train),(_,_) = cifar10.load_data()

    fig,ax = plt.subplots(nrows=10,ncols=10,figsize=(20,20))

    placeholder = st.empty()
    placeholder.text("Loading ...")
    
    for i in range(10):
        for j in range(10):
            ax[i,j].imshow(X_train[10*i+j])
            ax[i,j].set_title(f"Class : {classLabel[y_train[10*i+j][0]].capitalize()}")
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
        placeholder.text("Loading ...")
    st.pyplot(fig)
    placeholder.empty()

else:
    st.write("<h3>PREPARED BY</h3>",unsafe_allow_html=True)
    st.write("<p style='color:grey'>20BCE019 ARYAN AMLANI<br> \
    20BCE032 RAJDEEP BODAR<br>20BCE038 \
    PRATIK CHAUDHARY<br>20BCE046 DIVYRAJ CHUDASAMA \
    <br>20BCE073 KUNJ GANDHI</p>",unsafe_allow_html=True)
    st.write("")
    st.write("<h4>Institute of Technology, Nirma University</h4>",unsafe_allow_html=True)