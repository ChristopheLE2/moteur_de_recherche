from distances import *
from descripteur import *
import os 
from skimage.transform import resize 
import csv

def extractReqFeatures(fileName,algo_choice):  
    print(algo_choice)
    if fileName : 
        img = cv2.imread(fileName)
        resized_img = resize(img, (128*4, 64*4))
            
        if algo_choice==1: #Couleurs
            histB = cv2.calcHist([img],[0],None,[256],[0,256])
            histG = cv2.calcHist([img],[1],None,[256],[0,256])
            histR = cv2.calcHist([img],[2],None,[256],[0,256])
            vect_features = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)
        
        elif algo_choice==2: # Histo HSV
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            histH = cv2.calcHist([hsv],[0],None,[180],[0,180])
            histS = cv2.calcHist([hsv],[1],None,[256],[0,256])
            histV = cv2.calcHist([hsv],[2],None,[256],[0,256])
            vect_features = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)

        elif algo_choice==3: #SIFT
            sift = cv2.SIFT_create() #cv2.xfeatures2d.SIFT_create() pour py < 3.4 
            # Find the key point
            kps , vect_features = sift.detectAndCompute(img,None)
    
        elif algo_choice==4: #ORB
            orb = cv2.ORB_create()
            # finding key points and descriptors of both images using detectAndCompute() function
            key_point1,vect_features = orb.detectAndCompute(img,None)
			
        np.savetxt("Methode_"+str(algo_choice)+"_requete.txt" ,vect_features)
        print("saved")
        #print("vect_features", vect_features)
        return vect_features

def search_similar_images(filepath, descriptor, distance_metric, num_images):
    folder_model = './descriptors/' + descriptor
    features1 = []

    for j in os.listdir(folder_model):
        data = os.path.join(folder_model, j)
        if not data.endswith(".txt"):
            continue
        feature = np.loadtxt(data)
        features1.append((os.path.join('imageWang1000', os.path.basename(data).split('.')[0] + '.jpg'), feature))

    req = extractReqFeatures(filepath, descriptor)
    neighbors = getkVoisins(features1, req, num_images, distance_metric)
    results = [{'path': n[0], 'distance': n[1]} for n in neighbors]
    return results

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
            

def saveFeaturesToFiles(features,folderName, progressBar): 
    if not os.path.exists(folderName): 
       os.makedirs(folderName) 
    i=1 
    progressBar.setValue(0) 
    print(len(features)) 
    for name,feature in features: 
        print(i,"Writing",name) 
        _, numero = name.split('\\') 
        num_unique = numero.split('.')[0] 
        print(num_unique) 
        np.savetxt(folderName+"/"+num_unique+".txt",feature) 
        i+=1 
        progressBar.setValue(100*((i+1)/len(features))) 

def concatenate(filenames, models,folders, progressbar): 
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
            pas += 1 
            progressbar.setValue(int(100*((pas+1)/1000)))         
    saveFeaturesToFiles(feat, folder_name, progressbar) 
    return folder_name 