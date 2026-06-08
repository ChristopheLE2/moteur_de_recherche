import os
import time
import cv2
import numpy as np

INPUT_DIR = "/input"

def extractReqFeatures(fileName, algo_choice):
    if fileName:
        img = cv2.imread(fileName)
        if algo_choice == 3:  # SIFT
            sift = cv2.SIFT_create()
            kps, vect_features = sift.detectAndCompute(img, None)
        output_path = os.path.join(
            INPUT_DIR,
            "Methode_" + str(algo_choice) + "_requete.txt"
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