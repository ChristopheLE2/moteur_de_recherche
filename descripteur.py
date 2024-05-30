from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications import vgg16
from tensorflow.keras.applications import resnet50
from keras.applications import inception_v3
from keras.applications import mobilenet
from keras.applications import xception
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import numpy as np
import operator
import math
import os
import tensorflow as tf
import csv
import warnings
warnings.filterwarnings('ignore')
from keras.models import Model
import glob
import matplotlib.image as mpimg
import cv2

def generateHistogramme_HSV(filenames, progressBar):
    if not os.path.isdir("HSV"):
        os.mkdir("HSV")
    i=0
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        histH = cv2.calcHist([img],[0],None,[180],[0,180])
        histS = cv2.calcHist([img],[1],None,[256],[0,256])
        histV = cv2.calcHist([img],[2],None,[256],[0,256])
        feature = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)

        num_image, _ = path.split(".")
        np.savetxt("HSV/"+str(num_image)+".txt" ,feature)
        
        progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))
        i+=1
    print("indexation Hist HSV terminée !!!!")
        
def generateHistogramme_Color(filenames, progressBar):
    if not os.path.isdir("BGR"):
        os.mkdir("BGR")
    i=0
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        histB = cv2.calcHist([img],[0],None,[256],[0,256])
        histG = cv2.calcHist([img],[1],None,[256],[0,256])
        histR = cv2.calcHist([img],[2],None,[256],[0,256])
        feature = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)

        num_image, _ = path.split(".")
        np.savetxt("BGR/"+str(num_image)+".txt" ,feature)
        progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))
        i+=1
    print("indexation Hist Couleur terminée !!!!")

def generateSIFT(filenames, progressBar):
    if not os.path.isdir("SIFT"):
        os.mkdir("SIFT")
    i=0
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        featureSum = 0
        sift = cv2.SIFT_create()  
        kps , des = sift.detectAndCompute(img,None)

        num_image, _ = path.split(".")
        np.savetxt("SIFT/"+str(num_image)+".txt" ,des)
        progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))
        
        featureSum += len(kps)
        i+=1
    print("Indexation SIFT terminée !!!!")    


def generateORB(filenames, progressBar):
    if not os.path.isdir("ORB"):
        os.mkdir("ORB")
    i=0
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        orb = cv2.ORB_create()
        key_point1,descrip1 = orb.detectAndCompute(img,None)
        
        num_image, _ = path.split(".")
        np.savetxt("ORB/"+str(num_image)+".txt" ,descrip1 )
        progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))
        i+=1
    print("indexation ORB terminée !!!!")

def generateGLCM(filenames, progressBar): 
    if not os.path.isdir("GLCM"): 
        os.mkdir("GLCM") 
    distances=[1,-1] 
    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4] 
    i=0 
    for path in os.listdir(filenames): 
        image = cv2.imread(filenames+"/"+path) 
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
        gray = img_as_ubyte(gray) 
        glcmMatrix = greycomatrix(gray, distances=distances, angles=angles, normed=True) 
        glcmProperties1 = greycoprops(glcmMatrix,'contrast').ravel() 
        glcmProperties2 = greycoprops(glcmMatrix,'dissimilarity').ravel() 
        glcmProperties3 = greycoprops(glcmMatrix,'homogeneity').ravel() 
        glcmProperties4 = greycoprops(glcmMatrix,'energy').ravel() 
        glcmProperties5 = greycoprops(glcmMatrix,'correlation').ravel() 
        glcmProperties6 = greycoprops(glcmMatrix,'ASM').ravel() 
        feature = np.array([glcmProperties1,glcmProperties2,glcmProperties3,glcmProperties4,glcmProperties5,glcmProperties6]).ravel() 
        num_image, _ = path.split(".") 
        np.savetxt("GLCM/"+str(num_image)+".txt" ,feature) 
        progressBar.setValue(100*((i+1)/len(os.listdir(filenames)))) 
        i+=1 
    print("indexation GLCM terminée !!!!") 

def generateLBP(filenames, progressBar): 
    if not os.path.isdir("LBP"): 
        os.mkdir("LBP") 
    i=0 
    for path in os.listdir(filenames): 
        img = cv2.imread(filenames+"/"+path) 
        points=8 
        radius=1 
        method='default' 
        subSize=(70,70) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        img = cv2.resize(img,(350,350)) 
        fullLBPmatrix = local_binary_pattern(img,points,radius,method) 
        histograms = [] 
        for k in range(int(fullLBPmatrix.shape[0]/subSize[0])): 
            for j in range(int(fullLBPmatrix.shape[1]/subSize[1])): 
                subVector = fullLBPmatrix[k*subSize[0]:(k+1)*subSize[0],j*subSize[1]:(j+1)*subSize[1]].ravel() 
                subHist,edges = np.histogram(subVector,bins=int(2**points),range=(0,2**points)) 
                histograms = np.concatenate((histograms,subHist),axis=None) 
        num_image, _ = path.split(".") 
        np.savetxt("LBP/"+str(num_image)+".txt" ,histograms) 
        progressBar.setValue(100*((i+1)/len(os.listdir(filenames))))    
        i+=1 
    print("indexation LBP terminé !!!!")

def generateHOG(filenames, progressBar): 
    if not os.path.isdir("HOG"): 
        os.mkdir("HOG") 
    i=0 
    cellSize = (25,25) 
    blockSize = (50,50) 
    blockStride = (25,25) 
    nBins = 9 
    winSize = (350,350) 
    for path in os.listdir(filenames): 
        img = cv2.imread(filenames+"/"+path) 
        image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        image = cv2.resize(image,winSize) 
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nBins) 
        feature = hog.compute(image) 
        num_image, _ = path.split(".") 
        np.savetxt("HOG/"+str(num_image)+".txt" ,feature ) 
        progressBar.setValue(100*((i+1)/len(os.listdir(filenames)))) 
        i+=1 
    print("indexation HOG terminée !!!!") 


#VGG16
root = "" #Chemin vers la base d'images
features1 = []
big_folder="Features_train/" #Dossier pour stocker les caractéristiques
model = vgg16.VGG16(weights='imagenet', include_top=True)

if not os.path.exists(big_folder):
  os.makedirs(big_folder)
folder_model = os.path.join(big_folder, 'VGG16/')
if not os.path.exists(folder_model):
  os.makedirs(folder_model)


def indexation(output_file):
  pas =0
  for j in os.listdir(files) :
    data = os.path.join(files, j)
    print (data)
    if not data.endswith(".jpg"):
      continue
    file_name = os.path.basename(data) # load an image from file
    image = load_img(data, target_size=(224, 224))
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image) # predict the probability
    feature = np.array(feature[0])
    np.savetxt(folder_model+"/"+os.path.splitext(file_name)[0]+".txt",feature)
    features1.append((data,feature))
    print (pas)
    pas = pas+1
  with open(output_file, "w") as output:
    output.write(str(features1))
