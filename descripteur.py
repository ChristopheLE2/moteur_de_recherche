from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import resnet50
from keras.applications import inception_v3
from keras.applications import mobilenet
from keras.applications import xception

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
from skimage.feature import greycomatrix

from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
import numpy as np
import os
import cv2
import csv
import math
import time
import json
from json import JSONEncoder

files = 'dataset'
filesJson =  'Features'
#Descipteurs
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def writeJson(features, nameFileJson, exec_time, n_Images):
    imagesJson = {}

    imagesJson['index'] = {
        'nameDatabase' : "Voitures",
        'Descripteur' : nameFileJson,
        'exec_time' : exec_time,
        'n_Images' : n_Images
    }

    imagesJson['image'] = []

    for item in features:
          numpyData = {"array": item[1]}
          encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)

          imagesJson['image'].append({
            'data': item[0],
            'desc': encodedNumpyData,
          })

    with open(filesJson+nameFileJson+".json", "w") as output:
        print("Output in : \n "+ filesJson +nameFileJson+".json")
        json.dump(imagesJson, output)


def indexation(func, nameFileJson):
    features = [] # Liste pour stocker le nom de l'image et ses caractéristiques

    start_time = time.time()
    pas = 0
    listPath = os.listdir(files)
    listPath.sort()

    print(f'Indexation des images de la base de données {files} avec {str(func)} : \nStockage dans le fichier {nameFileJson}.json\n')

    for j in listPath :
        data = os.path.join(files, j)
        if not data.endswith(".jpg"):
            continue
        file_name = os.path.basename(data)
        # load an image from file
        image = cv2.imread(data)
        # describe image
        feature = func(image)
        features.append((file_name,feature))
        if pas%100 == 0:
            print (f'Nombres d\'images traitées : {pas+1}\nDernière image traitée : {file_name}\n\n')
        pas = pas + 1

    print('\nIndexation terminée, sauvegarde du fichier json')
    writeJson(features, nameFileJson, time.time()-start_time, pas)

    print('Temps d\'indexation : ', time.time() - start_time)

def concatenate(filenames, models,folders):
    algo_choix1=''
    algo_choix2=''
    nom=''
    folder_name1=''
    folder_name2=''
    pas=0
    feat=[]
    algo_choix1 =models[0]
    folder_model1 = folders[0]
    algo_choix2 = models[1]
    folder_model2 = folders[1]
    _,nom1 = folder_model1.split('/')
    _,nom2 = folder_model2.split('/')
    folder_name=nom1+"_"+nom2
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
        features=''
        data=''
        feature=''
        featureTOT=''
        for j in os.listdir(str(folder_model1)):
            data1=os.path.join(folder_model1,j)
            data2=os.path.join(folder_model2,j)
            feature1 = np.loadtxt(data1)
            feature2 = np.loadtxt(data2)
            if(algo_choix1 == 1 or algo_choix1 == 2):
                feature1 = feature1.ravel()
            if(algo_choix2 == 1 or algo_choix2 == 2):
                feature2 = feature2.ravel()
            featureTOT = np.concatenate([feature1,feature2])
            feat.append((os.path.join(filenames,os.path.basename(data1).split('.')[0]+'.jpg'),featureTOT))
    print('\nIndexation terminée, sauvegarde du fichier json')
    #writeJson(features, nameFileJson, time.time()-start_time, pas)
    return folder_name


def indexation_choix(descripteur, files):

    if type(descripteur) == list and len(descripteur) == 1:
        descripteur = descripteur[0]
        descriptor = {
            "HSV": HSV,
            "BGR": BGR,
            "SIFT": SIFT,
            "ORB": ORB,
            "LBP": LBP,
            "GLCM": GLCM,
            "HOG": HOG,
            "VGG16": indexationVGG16
        }

        if descripteur in descriptor:
            func = descriptor[descripteur]
            if descripteur == "VGG16":
                indexationVGG16(vgg16.VGG16, descripteur)
            else:
                indexation(func, descripteur)

    elif type(descripteur) and len(descripteur) == 2:
        folder_name = concatenate(files, descripteur, [f"Features_train/{descripteur[0]}", f"Features_train/{descripteur[1]}"])
        return f"indexation_{folder_name}.json"
    else:
        print("RIP")
        return None

#HSV
def HSV(image):
    hsv_featuresA = np.array([])
    if image is not None:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histH = cv2.calcHist([img], [0], None, [180], [0, 180])
        histS = cv2.calcHist([img], [1], None, [256], [0, 256])
        histV = cv2.calcHist([img], [2], None, [256], [0, 256])
        feature = np.concatenate((histH, np.concatenate((histS, histV), axis=None)), axis=None)
        hsv_featuresA = np.array(feature)
    return hsv_featuresA

#BGR

def BGR(image):
  hist_features = []
  if image is not None:
    histR = cv2.calcHist([image],[0],None,[256],[0,256])
    histG = cv2.calcHist([image],[1],None,[256],[0,256])
    histB = cv2.calcHist([image],[2],None,[256],[0,256])
    hist = [histR, histG, histB]
    hist_features.append(hist)
  hist_featuresA = np.array(hist_features)
  return hist_featuresA

#sift

def SIFT(image):

  if image is not None:
    sift = cv2.xfeatures2d.SIFT_create()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    keypoints = sift.detect(gray,None)
    keypoints_sift, descriptor_sift = sift.compute(gray, keypoints)
  if descriptor_sift is not None:
    sift_features = descriptor_sift.tolist()
    sift_featuresA = np.array(sift_features)
  else:
    sift_featuresA = np.array([-1])
    print('pas descriptor_sift')
  return sift_featuresA

#ORB

def ORB(image):
    orb_featuresA = np.array([-1])
    if image is not None:
        orb = cv2.ORB_create()
        kp = None
        des = None
        kp, des = orb.detectAndCompute(image, None)
        if des is not None:
            orb_featuresA = np.array(des)
        else:
            print('None des')
    return orb_featuresA

#GLCM
def GLCM(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2], symmetric=True, normed=True)
    return glcm

#LBP

def LBP(image):
  if image is not None:
    METHOD = 'uniform'
    radius = 3
    n_points = 8 * radius
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, METHOD)
    lbp_features = lbp.tolist()
  lbp_featuresA = np.array(lbp_features)
  return lbp_featuresA

#HOG

def HOG(image):
    cellSize = (25, 25)
    blockSize = (50, 50)
    blockStride = (25, 25)
    nBins = 9
    winSize = (350, 350)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, winSize)
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBins)
    feature = hog.compute(image)
    return feature

#VGG16
def indexationVGG16(files):
    root = "" #Chemin vers la base d'images
    features1 = []
    big_folder = "Features_train/" #Dossier pour stocker les caractéristiques
    model = vgg16.VGG16(weights='imagenet', include_top=True)

    if not os.path.exists(big_folder):
        os.makedirs(big_folder)
    folder_model = os.path.join(big_folder, 'VGG16/')
    if not os.path.exists(folder_model):
        os.makedirs(folder_model)

    pas = 0
    for j in os.listdir(files):
        data = os.path.join(files, j)
        print(data)
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
        np.savetxt(folder_model + "/" + os.path.splitext(file_name)[0] + ".txt", feature)
        features1.append((data, feature))
        print(pas)
        pas = pas + 1
    save_data("VGG16", files, features1, folder_model, None)
    print("Indexation VGG16 terminée !!!!")


#TEST

indexation_choix(['LBP'],files)
