from flask import Flask, render_template, request, jsonify

import os
import numpy as np
import cv2
import csv
import math
import time
import json
from json import JSONEncoder
import operator
from math import ceil

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
from skimage.feature import graycomatrix

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
from pathlib import Path


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def indexation(func, descripteur, nameFile):
    global features  
    features = []# Liste pour stocker le nom de l'image et ses caractéristiques

    #start = time.time() #pour nous quand on devait mesurer les performances en local, l'utilisateur n'a pas besoin de savoir cette information

    listPath = os.listdir(app.static_folder, descripteur, nameFile)
    listPath.sort()

    big_folder=os.path.join(app.static_folder,"Features/") #Dossier pour stocker les caractéristiques

    folder_model1=big_folder + nameFile + "/"
    if not os.path.exists(folder_model1):
      os.makedirs(folder_model1)


    data = os.path.join("./media/dataset", nameFile)
    file_name = os.path.basename(data)
    # charger une image du fichier
    image = cv2.imread(data)
    #description de l'image
    feature = func(image)
    features.append((file_name, feature))
    base_name = os.path.splitext(file_name)[0]
    np.savetxt(folder_model1 +base_name + ".txt", feature)

    #end = time.time()
    #print(f'Temps d\'indexation : { end - start:.2f} sec')

def concatenate(filenames, models, folders):
    global progress
    algo_choix1 = models[0]
    folder_model1 = folders[0]
    algo_choix2 = models[1]
    folder_model2 = folders[1]
    _, nom1 = folder_model1.split('/')
    _, nom2 = folder_model2.split('/')
    folder_name = nom1 + "_" + nom2

    if not os.path.isdir(folder_name):
        os.mkdir(app.static_folder,folder_name)

    features = []
    
    for j in os.listdir(str(folder_model1)):
        data1 = os.path.join(folder_model1, j)
        data2 = os.path.join(folder_model2, j)
        feature1 = np.loadtxt(data1)
        feature2 = np.loadtxt(data2)

        if algo_choix1 in ["HSV", "BGR", "SIFT", "ORB"]:
            feature1 = feature1.ravel()
        if algo_choix2 in ["HSV", "BGR", "SIFT", "ORB"]:
            feature2 = feature2.ravel()

        featureTOT = np.concatenate([feature1, feature2])
        features.append((os.path.join(filenames, os.path.basename(data1).split('.')[0] + '.jpg'), featureTOT))
        progress = 100*((j+1)/len(os.listdir(filenames)))

    saveFeaturesToFile(features, folder_name)
    return folder_name

def saveFeaturesToFile(features, folderName):
    if not os.path.exists(folderName):
        os.makedirs(app.static_folder,folderName)
    i = 1

    for name, feature in features:
        print(i, "Writing", name)
        _, numero = name.split('/')
        num_unique = numero.split('.')[0]
        print(num_unique)
        np.savetxt(folderName + "/" + num_unique + ".txt", feature)
        i += 1


def indexation_choix(descripteur):
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

    if isinstance(descripteur, list):
        if len(descripteur) == 1:
            descripteur = descripteur[0]
            if descripteur in descriptor:
                func = descriptor[descripteur]
                if descripteur == "VGG16":
                    indexationVGG16(fileName)
                else:
                    indexation(func, descripteur, fileName)

        elif len(descripteur) == 2:
            folder_name = concatenate(fileName, descripteur, [os.path.join("Features", descripteur[0]), os.path.join("Features", descripteur[1])])
            return f"indexation_{app.static_folder,folder_name}.txt"
        else:
            print("Nombre de descripteurs incorrect")
            return None

# Descripteurs
def HSV(image):
    if image is not None:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histH = cv2.calcHist([img], [0], None, [180], [0, 180])
        histS = cv2.calcHist([img], [1], None, [256], [0, 256])
        histV = cv2.calcHist([img], [2], None, [256], [0, 256])
        feature = np.concatenate((histH, np.concatenate((histS, histV), axis=None)), axis=None)

    return np.array(feature)

def BGR(image):
    if image is not None:
        histR = cv2.calcHist([image], [0], None, [256], [0, 256])
        histG = cv2.calcHist([image], [1], None, [256], [0, 256])
        histB = cv2.calcHist([image], [2], None, [256], [0, 256])
        feature = np.concatenate((histB, np.concatenate((histG, histR), axis=None)), axis=None)

    return np.array(feature)

def SIFT(image, scale_percent=50):
    if image is not None:
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptor_sift = sift.detectAndCompute(gray, None)
        
    return np.array(descriptor_sift)

def ORB(image):
    if image is not None:
        orb = cv2.ORB_create()
        kp = None
        des = None
        kp, des = orb.detectAndCompute(image, None)

    return np.array(des)

def GLCM(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2], symmetric=True, normed=True)
    return glcm

def LBP(image):
    if image is not None:
        METHOD = 'uniform'
        radius = 3
        n_points = 8 * radius
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, n_points, radius, METHOD)
        lbp_features = lbp.tolist()

    return np.array(lbp_features)

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

def indexationVGG16(files):
    features1 = []
    big_folder = app.static_folder + "Features"  # Dossier pour stocker les caractéristiques
    model1 = vgg16.VGG16(weights='imagenet', include_top=True)

    if not os.path.exists(big_folder):
        os.makedirs(big_folder)
    folder_model = os.path.join(big_folder, 'VGG16/')
    if not os.path.exists(folder_model):
        os.makedirs(folder_model)

    for j in os.listdir(files) :
        data = os.path.join(files, j)
        if not data.endswith(".jpg"):
            continue
        file_name = os.path.basename(data)
        # Chargement d'une image
        image = load_img(data, target_size=(224, 224))
        # Convertir les pixels d'une image en numpy array et reshape
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare l'image pour le modèle VGG
        image = preprocess_input(image)
        # prédit la proba de sortie
        feature = model1.predict(image)
        feature = np.array(feature[0])
        np.savetxt(big_folder + os.path.splitext(file_name)[0] + ".txt",feature)
        features1.append((data,feature))

    with open("Features_train/VGG16.txt", "w") as output:
        output.write(str(features1))

# Distances
def euclidean(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    distance = np.linalg.norm(l1 - l2)
    return distance


def chiSquareDistance(l1, l2):
    s = 0.0
    for i, j in zip(l1, l2):
        if np.all([i == 0, j == 0]):
            continue
        s += (i - j) ** 2 / (i + j)
    return s

def bhatta(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    num = np.sum(np.sqrt(np.multiply(l1, l2, dtype=np.float64)), dtype=np.float64)
    den = np.sqrt(np.sum(l1, dtype=np.float64) * np.sum(l2, dtype=np.float64))
    return math.sqrt(1 - num / den)

def flann(a, b):
    a = np.float32(np.array(a))
    b = np.float32(np.array(b))
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.inf
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flannMatcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = list(map(lambda x: x.distance, flannMatcher.match(a, b)))
    return np.mean(matches)

def bruteForceMatching(a, b):
    a = np.array(a).astype('uint8')
    b = np.array(b).astype('uint8')
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.inf
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = list(map(lambda x: x.distance, bf.match(a, b)))
    return np.mean(matches)

def distance_f(l1, l2, distanceName):
    if distanceName == "Euclidienne":
        distance = euclidean(l1, l2)
    elif distanceName == "Correlation":
        methode = cv2.HISTCMP_CORREL
        distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
    elif distanceName == "Chi carre":
        distance = chiSquareDistance(l1, l2)
    elif distanceName == "Intersection":
        methode = cv2.HISTCMP_INTERSECT
        distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
    elif distanceName == "Bhattacharyya":
        distance = bhatta(l1, l2)
    elif distanceName == "Brute force":
        distance = bruteForceMatching(l1, l2)
    elif distanceName == "Flann":
        distance = flann(l1, l2)
    return distance

def getkVoisins(lfeatures, req, k, distanceName):
    ldistances = []
    for i in range(len(lfeatures)):
        dist = distance_f(req, lfeatures[i][1], distanceName)
        ldistances.append((lfeatures[i][0], lfeatures[i][1], dist))
    if distanceName in ["Correlation", "Intersection"]:
        ordre = True
    else:
        ordre = False
    ldistances.sort(key=operator.itemgetter(2), reverse=ordre)

    lvoisins = []
    for i in range(k):
        lvoisins.append(ldistances[i])
    return lvoisins

def find_index_by_filename(filename, features):
    for index, feature in enumerate(features):
        if os.path.basename(feature[0]) == filename:
            return index
    return None

def recherche(index, features, distance_name, top=10):
    #start = time.time() #pour nous quand on devait mesurer les performances en local, l'utilisateur n'a pas besoin de savoir cette information

    # Générer les voisins
    voisins = getkVoisins(features, features[index][1], top, distance_name)

    #end = time.time()
    #print(f"Time for Recherche: {end - start:.2f} seconds")

    path_image_plus_proches = [voisin[0] for voisin in voisins]
    nom_image_plus_proches = [os.path.basename(voisin[0]) for voisin in voisins]

    return nom_image_plus_proches

def calculate_metrics(nom_image_plus_proches, fileName, sortie, total_pertinent):
    rappel_precision = []
    rappels = []
    precisions = []
    filename_req = os.path.basename(app.static_folder, fileName)
    num_image = filename_req.split("_")[-1].split(".")[0]
    classe_image_requete = int(num_image) // 100
    val = 0

    #start = time.time()   #pour nous quand on devait mesurer les performances en local, l'utilisateur n'a pas besoin de savoir cette information
    for j in range(sortie):
        num_image_proche = nom_image_plus_proches[j].split("_")[-1].split(".")[0]
        classe_image_proche = int(num_image_proche) // 100
        classe_image_requete = int(classe_image_requete)
        classe_image_proche = int(classe_image_proche)
        if classe_image_requete == classe_image_proche:
            rappel_precision.append(True) # Bonne classe (pertinant)
            val += 1
        else:
            rappel_precision.append(False) # Mauvaise classe (non pertinant)
        #print(f"Image proche: {nom_image_plus_proches[j]}, Classe: {classe_image_proche}, Pertinent: {rappel_precision[-1]}")

    for i in range(sortie):
        j = i
        val = 0
        while j >= 0:
            if rappel_precision[j]:
                val += 1
            j -= 1
        precision = val / (i + 1)
        rappel = val / total_pertinent
        rappels.append(rappel)
        precisions.append(precision)

    #end = time.time()
    #print(f"Rappels: {rappels}")
    #print(f"Precisions: {precisions}")

    # Calcul des métriques supplémentaires
    AP = np.mean([precisions[i] for i in range(sortie) if rappel_precision[i]])
    MAP = np.mean(AP)
    R_Precision = precisions[total_pertinent-1] if total_pertinent <= sortie else precisions[-1]

    #print(f"Average Precision (AP): {AP}")
    #print(f"Mean Average Precision (MAP): {MAP}")
    #print(f"R-Precision: {R_Precision}")

    # Création de la courbe R/P
    plt.figure(figsize=(12, 8))  # Ajustez la taille de la figure ici
    plt.plot(rappels, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"R/P {sortie} voisins de l'image n°{num_image}")

    # Enregistrement de la courbe RP
    save_folder = os.path.join(".", num_image)
    if not os.path.exists(save_folder):
        os.makedirs(app.static_folder, save_folder)
    save_name = os.path.join(save_folder, num_image + '.png')
    plt.savefig(os.path.join(app.static_folder,f'{nom_image_plus_proches}'), format='png', dpi=600)
    plt.close()


@app.route('/get-images', methods=['GET', 'POST'])
def get_images():
    global noms_descripteur, distance_name, top, fileName

    matplotlib.use('agg')

    noms_descripteur = request.form.get('descriptor_type')
    distance_name = request.form.get('similarity_calculation')
    top = request.form.get('images_to_display')
    fileName = f"/dataset/{request.files['fileInput'].filename}"
    
    index = find_index_by_filename(fileName, features)

    nom_image_plus_proches = recherche(index, features, distance_name, top)

    num_image = fileName.split("_")[-1].split(".")[0]
    total_pertinent = sum(1 for feature in features if int(os.path.basename(feature[0]).split("_")[-1].split(".")[0]) // 100 == int(num_image) // 100)
    calculate_metrics(nom_image_plus_proches, fileName, top, total_pertinent)

    return jsonify({'img_to_show', nom_image_plus_proches} )

@app.route('/progress')
def get_progress():
    global progress
    progress = 0
    return jsonify({"progress": progress})


@app.route('/nosInfos')
def showNosInfos():
    return render_template('nosInfos.html')

if __name__ == '__main__':
    app.run(debug=True)