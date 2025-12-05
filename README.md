# README v2 – Intégration de nouvelles données & migration du schéma FastIA

## 🎯 Objectif du projet
Cette seconde phase du projet FastIA vise à :
1. Analyser de nouvelles données socio-démographiques et économiques  
2. Nettoyer et transformer ces données  
3. Faire évoluer la base relationnelle existante  
4. Appliquer une migration Alembic  
5. Conserver la compatibilité avec l’API  
6. Mettre à jour le pipeline IA pour inclure les nouvelles variables pertinentes

---

## 🗂️ Structure du projet
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
│  ├─ load_data.py
│  └─ load_data_v2.py
│
├─ migrations/
│   └─ versions/
│       └─ X_add_socio_demo.py
│
├─ data/
│  └─ data-all-complete.csv
│
├─ artifacts/
│  ├─ credit_score_model_weights.npz
│  └─ loss_curve.png
│
├─ fastia.db
└─ README_v2.md
```

---

## 🧪 Analyse des nouvelles données
Les colonnes ajoutées :
- `orientation_sexuelle` (donnée sensible)
- `nb_enfants`
- `quotient_caf`

Problèmes identifiés :
- valeurs manquantes dans certaines colonnes
- incohérences (nb_enfants négatifs)
- outliers dans quotient_caf
- données éthiquement sensibles

Actions menées :
- normalisation des types
- correction des outliers
- exclusion éthique de `orientation_sexuelle` du modèle IA

---

## 🗃️ Migration Alembic
Une migration a été créée pour ajouter les colonnes :

```
orientation_sexuelle : String(20)
nb_enfants : Integer
quotient_caf : Float
```

Commande pour appliquer la migration :

```
alembic upgrade head
```

---

## 🧼 Pipeline d’ingestion v2
Le script `load_data_v2.py` :
- nettoie les colonnes
- corrige les valeurs aberrantes
- filtre les lignes trop incomplètes
- insère les données dans la table clients + financial_info

Lancement :

```
python -m scripts.load_data_v2
```

---

## 🤖 Mise à jour du modèle IA
Le modèle IA inclut maintenant :
- nb_enfants
- quotient_caf  

💡 orientation_sexuelle est volontairement exclue

Lancement de l’entraînement :

```
python -m app.ml.train_model
```

Résultats générés dans `artifacts/` :
- `credit_score_model_weights.npz`
- `loss_curve.png`

---

## 🔐 Analyse éthique
- exclusion de données sensibles (orientation sexuelle)
- risques de biais socio-économiques documentés
- pipeline reproductible et transparent

---

## ✅ Conclusion
Le système est désormais :
- étendu  
- migré proprement  
- compatible avec l’API existante  
- documenté techniquement et éthiquement  
