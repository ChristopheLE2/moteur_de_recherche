# 🔍 Moteur de Recherche d’Images

**Comparaison de descripteurs visuels (SIFT, ORB, HOG, BGR, HSV) avec interface web & déploiement Docker**

---

## 📌 Présentation

Ce projet implémente un **moteur de recherche d’images basé sur la similarité visuelle**, utilisant plusieurs familles de descripteurs :

- **Descripteurs locaux** : SIFT, ORB  
- **Descripteurs globaux** : HOG, histogrammes BGR, histogrammes HSV  
- **Descripteurs hybrides** : HOG+HSV, BGR+HSV  

Une interface web permet d’**uploader une image**, de choisir un descripteur, et d’obtenir les **images les plus similaires**.  
Le tout est servi via **Flask**, **Nginx** et **Docker**.

---

## 🚀 Fonctionnalités

- Upload d’image via interface web  
- Extraction de features selon plusieurs méthodes  
- Recherche par similarité (distance euclidienne / cosinus)  
- Visualisation des résultats  
- Déploiement complet via Docker + Nginx  
- Pages HTML personnalisées (index + équipe)

---

## 🏗️ Architecture du projet

```
moteur_de_recherche/
├── docker-compose.yml
├── Dockerfile
├── main.py
├── nginx.conf
├── README.md
├── requirements.txt
├── templates/
│   ├── index.html
│   └── nosInfos.html
└── static/
    ├── bibli.jpg
    ├── christophe.png
    ├── hugo.png
    ├── zoe.png
    ├── SIFT.zip
    ├── styles/
    │   ├── styles_main.css
    │   └── styles_nosInfos.css
    ├── js/
    │   └── script.js
    ├── BGR/
    ├── BGR_HSV/
    ├── HOG/
    ├── HOG_HSV/
    └── ORB/
```

---

## 🛠️ Technologies utilisées

| Domaine | Outils |
|--------|--------|
| Backend | Python, Flask |
| Vision | OpenCV, NumPy, Scikit-image |
| Web | HTML, CSS, JS |
| Déploiement | Docker, Nginx |
| Données | Descripteurs pré-calculés (SIFT, ORB, HOG, BGR, HSV) |

---

## 📦 Installation & exécution

### 1. Cloner le dépôt
```bash
git clone https://github.com/<user>/<repo>.git
cd moteur_de_recherche-main
```

### 2. Lancer avec Docker (recommandé)
```bash
docker-compose up --build
```

L’application sera disponible sur :  
👉 http://localhost:8080

### 3. Exécution locale (sans Docker)
```bash
pip install -r requirements.txt
python main.py
```

---

## 🧠 Comment ça marche ?

### 1. Extraction des descripteurs  
Selon la méthode choisie, l’image est transformée en :

- vecteurs de points clés (SIFT, ORB)
- histogrammes de couleurs (BGR, HSV)
- histogrammes de gradients (HOG)
- combinaisons couleur + forme

### 2. Chargement des descripteurs pré-calculés  
Les dossiers `ORB/`, `HOG/`, `BGR/`, etc. contiennent les features des images de référence.

### 3. Calcul de similarité  
Comparaison via :

- distance euclidienne  
- distance cosinus  

### 4. Retour des images les plus proches  
Tri et affichage dans l’interface web.

---

## 👥 Équipe

- **Hugo** — Vision par ordinateur, backend, optimisation  
- **Christophe** — Interface web, intégration  
- **Zoé** — Design, UX, documentation  

---

## 📌 Améliorations possibles

- [ ] Ajout d’un index FAISS pour accélérer la recherche  
- [ ] Ajout de descripteurs deep learning (ResNet, CLIP)  
- [ ] Support du drag & drop  
- [ ] Visualisation interactive des features  
- [ ] API REST complète  

---

## 📄 Licence

Ce projet est distribué sous licence MIT.  
Vous êtes libres de l’utiliser, le modifier et le redistribuer.

---

## ⭐ Contribuer

Les contributions sont les bienvenues !  
N’hésitez pas à ouvrir une *issue* ou une *pull request*.
