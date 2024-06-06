from flask import Flask, render_template, request, jsonify
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb
from skimage.feature import graycomatrix
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
import operator
import os

app = Flask(__name__)
progress = 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    # Vous pouvez enregistrer le fichier ou faire quelque chose avec ici
    return f'File {file.filename} uploaded successfully'


def euclidean(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    distance = np.linalg.norm(l1 - l2)
    return distance

def chiSquareDistance(l1, l2):
    s = 0.0
    for i, j in zip(l1, l2):
        if i == j == 0.0:
            continue
        s += (i - j)**2 / (i + j)
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
    sch_params = dict(checks=50)
    flannMatcher = cv2.FlannBasedMatcher(index_params, sch_params)
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
    elif distanceName in ["Correlation", "Chi carre", "Intersection", "Bhattacharyya"]:
        if distanceName == "Correlation":
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

# Chemins pour stocker les descripteurs
DESCRIPTOR_PATHS = {
    "HSV": "HSV",
    "BGR": "BGR",
    "SIFT": "SIFT",
    "ORB": "ORB"
}

# Fonction pour créer les répertoires si nécessaires
def create_descriptor_dirs():
    for path in DESCRIPTOR_PATHS.values():
        if not os.path.exists(path):
            os.makedirs(path)

# Fonction d'indexation pour histogramme HSV
def generate_histogramme_hsv(filenames):
    global progress
    progress = 0
    i = 0

    for path in os.listdir(filenames):
        img = cv2.imread(os.path.join(filenames, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        histH = cv2.calcHist([img], [0], None, [180], [0, 180])
        histS = cv2.calcHist([img], [1], None, [256], [0, 256])
        histV = cv2.calcHist([img], [2], None, [256], [0, 256])
        feature = np.concatenate((histH, np.concatenate((histS, histV), axis=None)), axis=None)
        num_image = os.path.splitext(path)[0]
        np.savetxt(os.path.join(DESCRIPTOR_PATHS["HSV"], f"{num_image}.txt"), feature)
        progress = 100*((i+1)/len(os.listdir(filenames)))
        i+=1

# Fonction d'indexation pour histogramme BGR
def generate_histogramme_bgr(filenames):
    global progress
    progress = 0
    i = 0

    for path in os.listdir(filenames):
        img = cv2.imread(os.path.join(filenames, path))
        histB = cv2.calcHist([img], [0], None, [256], [0, 256])
        histG = cv2.calcHist([img], [1], None, [256], [0, 256])
        histR = cv2.calcHist([img], [2], None, [256], [0, 256])
        feature = np.concatenate((histB, np.concatenate((histG, histR), axis=None)), axis=None)
        num_image = os.path.splitext(path)[0]
        np.savetxt(os.path.join(DESCRIPTOR_PATHS["BGR"], f"{num_image}.txt"), feature)
        progress = 100*((i+1)/len(os.listdir(filenames)))
        i+=1

# Fonction d'indexation pour SIFT
def generate_sift(filenames):
    global progress
    progress = 0
    i = 0

    sift = cv2.SIFT_create()
    for path in os.listdir(filenames):
        img = cv2.imread(os.path.join(filenames, path))
        kps, des = sift.detectAndCompute(img, None)
        num_image = os.path.splitext(path)[0]
        np.savetxt(os.path.join(DESCRIPTOR_PATHS["SIFT"], f"{num_image}.txt"), des)
        progress = 100*((i+1)/len(os.listdir(filenames)))
        i+=1

# Fonction d'indexation pour ORB
def generate_orb(filenames):
    global progress
    progress = 0
    i = 0

    orb = cv2.ORB_create()
    for path in os.listdir(filenames):
        img = cv2.imread(os.path.join(filenames, path))
        key_point1, descrip1 = orb.detectAndCompute(img, None)
        num_image = os.path.splitext(path)[0]
        np.savetxt(os.path.join(DESCRIPTOR_PATHS["ORB"], f"{num_image}.txt"), descrip1)
        progress = 100*((i+1)/len(os.listdir(filenames)))
        i+=1

def main_indexing(filenames):
    create_descriptor_dirs()
    generate_histogramme_hsv(filenames)
    generate_histogramme_bgr(filenames)
    generate_sift(filenames)
    generate_orb(filenames)
    print("Indexation terminée!")


def get_k_neighbors(lfeatures, req, k, distance_name):
    ldistances = []
    for feature in lfeatures:
        dist = distance_f(req, feature[1], distance_name)
        ldistances.append((feature[0], feature[1], dist))

    if distance_name in ["Correlation", "Intersection"]:
        order = True
    else:
        order = False

def extract_req_features(file_name, algo_choice):
    img = cv2.imread(file_name)
    if algo_choice == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        histH = cv2.calcHist([img], [0], None, [180], [0, 180])
        histS = cv2.calcHist([img], [1], None, [256], [0, 256])
        histV = cv2.calcHist([img], [2], None, [256], [0, 256])
        return np.concatenate((histH, np.concatenate((histS, histV), axis=None)), axis=None)
    elif algo_choice == "BGR":
        histB = cv2.calcHist([img], [0], None, [256], [0, 256])
        histG = cv2.calcHist([img], [1], None, [256], [0, 256])
        histR = cv2.calcHist([img], [2], None, [256], [0, 256])
        return np.concatenate((histB, np.concatenate((histG, histR), axis=None)), axis=None)
    elif algo_choice == "SIFT":
        sift = cv2.SIFT_create()
        kps, des = sift.detectAndCompute(img, None)
        return des
    elif algo_choice == "ORB":
        orb = cv2.ORB_create()
        key_point1, descrip1 = orb.detectAndCompute(img, None)
        return descrip1
    return None

def main_search(query_image, algo_choice, distance_name, k=20):
    # Charger les caractéristiques indexées
    lfeatures = []
    for file in os.listdir(DESCRIPTOR_PATHS[algo_choice]):
        if file.endswith(".txt"):
            image_name = file.split(".")[0]
            feature_vector = np.loadtxt(os.path.join(DESCRIPTOR_PATHS[algo_choice], file))
            lfeatures.append((image_name, feature_vector))

    # Extraction des caractéristiques de l'image de requête
    query_features = extract_req_features(query_image, algo_choice)

    # Recherche des k plus proches voisins
    neighbors = get_k_neighbors(lfeatures, query_features, k, distance_name)

    return neighbors

def find_index_by_filename(filename, directory):
    for index, file in enumerate(os.listdir(directory)):
        if file.endswith('.jpg') and file == filename:
            return index
    return None

def Compute_RP(RP_file, top,nom_image_requete, nom_images_non_proches):
  text_file = open(RP_file, "w")
  rappel_precision=[]
  rp = []
  position1=int(nom_image_requete)//100
  for j in range(top):
    position2=int(nom_images_non_proches[j])//100
    if position1==position2:
      rappel_precision.append("pertinant")
    else:
      rappel_precision.append("non pertinant")
  for i in range(top):
    j=i
    val=0
    while j>=0:
      if rappel_precision[j]=="pertinant":
        val+=1
      j-=1
    rp.append(str((val/(i+1))*100)+" "+str((val/top)*100))
  with open(RP_file, 'w') as s:
    for a in rp:
      s.write(str(a) + '\n')
  print(rp)

@app.route('/show_RP', methods=['GET'])
def Display_RP(fichier):
  x = []
  y = []
  with open(fichier) as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
      x.append(float(row[0]))
      y.append(float(row[1]))
      fig = plt.figure()
  plt.plot(y,x,'C1', label="VGG16" );
  plt.xlabel('Rappel')
  plt.ylabel('Précison')
  plt.title("R/P")
  plt.legend()
  plt.savefig(os.path.join(app.static_folder,'new_plot.png'))
  return jsonify(os.path.join(app.static_folder,'new_plot.png'))

@app.route('/show_test', methods=['GET'])
def display():
    lnprice=np.log(15)
    plt.plot(lnprice)
    plt.savefig(os.path.join(app.static_folder,'new_plot.png'))
    return jsonify(os.path.join(app.static_folder,'new_plot.png'))






@app.route('/get-images', methods=['GET'])
def get_images():
    images_folder = os.path.join(app.static_folder)
    image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
    return jsonify(image_files)

@app.route('/progress')
def get_progress():
    global progress
    return jsonify({"progress": progress})



@app.route('/nosInfos')
def showNosInfos():
    return render_template('nosInfos.html')

if __name__ == '__main__':
    app.run(debug=True)