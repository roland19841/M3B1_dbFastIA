# FastIA – API REST & Modèle IA sur Base de Données Relationnelle
Projet Module 3 Brief 1 – Formation IA – OPCO ATLAS

## 🎯 Objectif du projet
Ce projet consiste à :

1. Créer une base de données relationnelle (SQLite) à partir des données du module 2.  
2. Exposer ces données via une API REST avec FastAPI + SQLAlchemy.  
3. Entraîner un modèle IA (réseau de neurones NumPy) à partir des données importées.  
4. Générer :
   - une documentation Swagger
   - une courbe de loss (train/validation)
   - les poids du modèle
   - un projet organisé proprement (routes, modèles, CRUD…)

## 📂 Structure du projet
```
fastia_project/
│
├─ app/
│  ├─ main.py
│  ├─ database.py
│  ├─ models.py
│  ├─ schemas.py
│  ├─ crud.py
│  ├─ routers/
│  │   └─ clients.py
│  └─ ml/
│      └─ train_model.py
│
├─ scripts/
│  └─ load_data.py
│
├─ data/
│  └─ data-all.csv
│
├─ artifacts/
│  ├─ credit_score_model_weights.npz
│  └─ loss_curve.png
│
├─ fastia.db
├─ requirements.txt
└─ README.md
```

## 🚀 Installation & démarrage

### 1️⃣ Créer un environnement Python 3.11
```
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2️⃣ Installer les dépendances
```
pip install --upgrade pip
pip install -r requirements.txt
```

## 🛠️ Chargement des données
```
python -m scripts.load_data
```

## 🌐 Lancer l’API FastAPI + Swagger
```
uvicorn app.main:app --reload
```

Swagger : http://127.0.0.1:8000/docs  
Redoc : http://127.0.0.1:8000/redoc

## 🧠 Entraîner le modèle IA
```
python -m app.ml.train_model
```

Résultats générés dans `artifacts/`.

## 📦 Livrables
- Modèles ORM  
- API FastAPI fonctionnelle  
- Routes GET / POST / DELETE  
- Documentation Swagger  
- Script d'import  
- Modèle IA + courbe de loss  
- Poids du modèle  
- README complet  
