import os
import time
import cv2
import numpy as np

INPUT_DIR = "/input"

def extractReqFeatures(fileName, algo_choice):
    if fileName:
        img = cv2.imread(fileName)

        if algo_choice==1: #Couleurs
            algo_name = "BGR"
            histB = cv2.calcHist([img],[0],None,[256],[0,256])
            histG = cv2.calcHist([img],[1],None,[256],[0,256])
            histR = cv2.calcHist([img],[2],None,[256],[0,256])
            vect_features = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)
        
        elif algo_choice==2: # Histo HSV
            algo_name = "HSV"
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            histH = cv2.calcHist([hsv],[0],None,[256],[0,256])
            histS = cv2.calcHist([hsv],[1],None,[256],[0,256])
            histV = cv2.calcHist([hsv],[2],None,[256],[0,256])
            vect_features = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)

        if algo_choice == 3:  # SIFT
            algo_name = "SIFT"
            sift = cv2.SIFT_create()
            kps, vect_features = sift.detectAndCompute(img, None)
        
        elif algo_choice==4: #ORB
            algo_name = "ORB"
            orb = cv2.ORB_create()
            # finding key points and descriptors of both images using detectAndCompute() function
            key_point1,vect_features = orb.detectAndCompute(img,None)
        
        
        output_path = os.path.join(
            INPUT_DIR,
            "Methode_" + str(algo_name) + "_requete.txt"
        )

        np.savetxt(output_path, vect_features)
        print("saved in", output_path)
        return vect_features

while True:
    job_file = os.path.join(INPUT_DIR, "process.job")
    if os.path.exists(job_file):
        image_path = os.path.join(INPUT_DIR, "image.jpg")
        try:
            extractReqFeatures(image_path, 3)
            open(
                os.path.join(INPUT_DIR, "done.flag"),
                "w"
            ).close()
        except Exception as e:
            print("Erreur :", e)
        os.remove(job_file)

    time.sleep(1)