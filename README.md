# Projet FastIA – Mise à jour du modèle IA avec adaptation de la couche d’entrée

## 📌 Contexte général
FastIA a enrichi sa base de données avec de nouvelles variables (`nb_enfants`, `quotient_caf`).  
L’ancien modèle IA ne pouvait utiliser que l'ancien schéma, il était donc nécessaire :

- d’adapter **uniquement la couche d’entrée** du modèle existant,
- de **conserver tous les poids internes appris** (couches cachées),
- de **réentraîner** le modèle sur le nouveau schéma,
- et d’**exposer** ce modèle mis à jour via une API FastAPI.

Ce README décrit les choix, les scripts, le pipeline complet, les performances observées et l’usage de TensorBoard.

---

# 1. Architecture globale du projet

```
data/                   → fichiers CSV sources
app/
 ├── ml/
 │    ├── tf_train_v1.py          → entraînement modèle v1 (ancien schéma)
 │    ├── tf_adapt_to_v2.py       → adaptation du modèle : élargissement de la couche d’entrée
 │    ├── tf_train_v2.py          → réentraînement du modèle adapté
 │
 ├── routers/
 │      ├── predict.py            → API FastAPI exposant POST /predict
 │
 ├── main.py                      → point d'entrée FastAPI
 ├── database.py, models.py       → ORM SQLAlchemy
artifacts/
 ├── model_v1_old_schema.h5       → modèle initial
 ├── model_v2_new_schema_init.h5  → modèle adapté non réentraîné
 ├── model_v2_new_schema.h5       → modèle final entraîné
 ├── encoder_v1.pkl               → encoder catégoriel (v1)
 ├── scaler_v1.pkl                → scaler anciennes features
 ├── extra_scaler_v2.pkl          → scaler nouvelles features
logs/
 ├── v1/                          → logs TensorBoard modèle v1
 ├── v2/                          → logs TensorBoard modèle v2
```

---

# 2. Étape 1 – Entraînement du modèle V1 (ancien schéma)

## Caractéristiques du modèle v1
- Entrée : 33 features (numériques + catégorielles encodées)
- Architecture :
  - Dense(32, relu)
  - Dense(16, relu)
  - Dense(1)
- Optimiseur : Adam, lr=1e-3
- EarlyStopping + TensorBoard

## Script : `tf_train_v1.py`
Ce script :

1. charge les données depuis la base SQLite,
2. encode les variables catégorielles (`encoder_v1.pkl`),
3. scale les features (`scaler_v1.pkl`),
4. entraîne le modèle,
5. produit la courbe de loss,
6. sauvegarde le modèle.

## 📊 Résultat du modèle V1

```
[V1] MSE validation finale : 26642.6758
```

Évaluation calculée **après restauration des meilleurs poids** via EarlyStopping.

---

# 3. Étape 2 – Adaptation de la couche d’entrée (modèle V2 initial)

## Objectif
- Ajouter **2 nouvelles colonnes** : `nb_enfants`, `quotient_caf`
- Conserver **tous les poids internes de v1**
- N’augmenter que la dimension d’entrée

## Méthode
1. Charger le modèle v1 **sans recompiler**.
2. Lire W_old (poids de la première Dense) → dimension `(33, 32)`
3. Construire W_new → dimension `(35, 32)` :
   - recopier les 33 lignes existantes,
   - initialiser les 2 nouvelles lignes avec une distribution faible `N(0, 0.01)`
4. Copier les poids des autres couches **à l’identique**.
5. Sauvegarder : `model_v2_new_schema_init.h5`.

---

# 4. Étape 3 – Entraînement du modèle V2 (nouveau schéma étendu)

## Pipeline
Le script `tf_train_v2.py` :

1. recharge `encoder_v1.pkl` + `scaler_v1.pkl`,
2. applique la transformation **exactement comme v1** pour les anciennes colonnes,
3. ajoute un **nouveau scaler** pour les nouvelles features (`extra_scaler_v2.pkl`),
4. concatène `[X_old_scaled || new_scaled]`,
5. recharge `model_v2_new_schema_init.h5`,
6. entraîne le modèle,
7. logge dans `logs/v2`.

## 📊 Résultat du modèle V2

```
[V2] MSE validation finale : 25663.3750
```

---

# 5. Analyse comparative des performances

| Modèle | Schéma | MSE Validation |
|--------|--------|----------------|
| **V1** | ancien (33 features) | **26642.6758** |
| **V2** | étendu (35 features) | **25663.3750** |

### 📈 Conclusion
- Le modèle V2 **réduit l’erreur de validation d’environ 3,7%**  
  → (26642 → 25663)
- Les nouvelles variables `nb_enfants` et `quotient_caf` apportent une **légère valeur ajoutée**.
- Les poids internes hérités de V1 ont permis :
  - un entraînement plus rapide,
  - une stabilisation immédiate de la loss,
  - un comportement cohérent du réseau.

---

# 6. Étape 4 – API FastAPI : exposition du modèle

## Route principale
Méthode : `POST /predict`

Corps attendu :

```json
{
  "age": 40,
  "height_cm": 175,
  "weight_kg": 80,
  "monthly_income": 2500,
  "credit_history": 3,
  "personal_risk": 0.4,
  "monthly_rent": 800,
  "loan_amount": 150000,
  "sex": "H",
  "sport_licence": "oui",
  "education_level": "licence",
  "region": "Île-de-France",
  "smoker": "non",
  "is_french": "oui",
  "family_status": "marié",
  "nb_enfants": 2,
  "quotient_caf": 750
}
```

Réponse :

```json
{
  "predicted_credit_score": 12345.67
}
```

---

# 7. Lancer TensorBoard

```bash
tensorboard --logdir logs
```

Accès :  
👉 http://localhost:6006/

---

# 8. Lancer l’API

```bash
uvicorn app.main:app --reload
```

Swagger UI :  
👉 http://127.0.0.1:8000/docs

---

# 9. Conclusion générale

- L'approche “**adaptation structurelle**” permet d’étendre un modèle tout en préservant son apprentissage.  
- L'héritage des poids internes a permis une **meilleure stabilité** et un **temps d'entraînement réduit**.  
- Les nouvelles variables ont un impact positif mais modéré, suggérant un potentiel pour :
  - augmenter la profondeur du réseau,
  - appliquer une meilleure sélection de features,
  - tester un modèle non linéaire plus puissant.

---

# 10. Commandes de reproductibilité

## Entraîner v1
```bash
python -m app.ml.tf_train_v1
```

## Adapter vers v2
```bash
python -m app.ml.tf_adapt_to_v2
```

## Entraîner v2
```bash
python -m app.ml.tf_train_v2
```

## Lancer l’API
```bash
uvicorn app.main:app --reload
```

---

# 11. Licence
Projet pédagogique – FastIA, Module IA & Industrialisation.
