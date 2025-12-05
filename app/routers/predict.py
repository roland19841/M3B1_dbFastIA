from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
import tensorflow as tf

# Schéma d'entrée pour la prédiction
class ClientFeatures(BaseModel):
    age: int
    height_cm: float
    weight_kg: float
    monthly_income: float
    credit_history: float
    personal_risk: float
    monthly_rent: float
    loan_amount: float

    sex: str
    sport_licence: str
    education_level: str
    region: str
    smoker: str
    is_french: str
    family_status: str

    nb_enfants: int
    quotient_caf: float


router = APIRouter(prefix="/predict", tags=["prediction"])

# Chargement des artefacts au démarrage du module
MODEL_PATH = os.path.join("artifacts", "model_v2_new_schema.h5")
ENCODER_PATH = os.path.join("artifacts", "encoder_v1.pkl")
SCALER_OLD_PATH = os.path.join("artifacts", "scaler_v1.pkl")
EXTRA_SCALER_PATH = os.path.join("artifacts", "extra_scaler_v2.pkl")

if not (os.path.exists(MODEL_PATH)
        and os.path.exists(ENCODER_PATH)
        and os.path.exists(SCALER_OLD_PATH)
        and os.path.exists(EXTRA_SCALER_PATH)):
    raise RuntimeError(
        "Modèle ou artefacts de prétraitement manquants. "
        "Assurez-vous d'avoir exécuté tf_train_v1, tf_adapt_to_v2 et tf_train_v2."
    )

model_v2 = tf.keras.models.load_model(MODEL_PATH, compile=False)
encoder_v1 = joblib.load(ENCODER_PATH)
scaler_old = joblib.load(SCALER_OLD_PATH)
extra_scaler = joblib.load(EXTRA_SCALER_PATH)


def preprocess_features(features: ClientFeatures) -> np.ndarray:
    """
    Recrée exactement le pipeline de features utilisé pour l'entraînement v2 :
    - numériques anciennes → concaténées aux catégorielles encodées → scaler_OLD
    - nouvelles features (nb_enfants, quotient_caf) → extra_scaler
    - concat [X_old_scaled || extra_scaled]
    """
    # Numériques anciennes
    numeric_old = np.array([[
        features.age,
        features.height_cm,
        features.weight_kg,
        features.monthly_income,
        features.credit_history,
        features.personal_risk,
        features.monthly_rent,
        features.loan_amount,
    ]], dtype=np.float32)

    # Catégorielles
    cat = [[
        features.sex,
        features.sport_licence,
        features.education_level,
        features.region,
        features.smoker,
        features.is_french,
        features.family_status,
    ]]

    cat_encoded = encoder_v1.transform(cat)

    X_old = np.concatenate([numeric_old, cat_encoded], axis=1)
    X_old_scaled = scaler_old.transform(X_old)

    # Nouvelles features
    extra_numeric = np.array([[
        features.nb_enfants,
        features.quotient_caf,
    ]], dtype=np.float32)

    extra_scaled = extra_scaler.transform(extra_numeric)

    X_full = np.concatenate([X_old_scaled, extra_scaled], axis=1)
    return X_full


@router.post("/", summary="Prédiction de score de crédit")
def predict_credit_score(features: ClientFeatures):
    try:
        X = preprocess_features(features)
        y_pred = model_v2.predict(X)
        score = float(y_pred[0, 0])
        return {
            "predicted_credit_score": score,
            "details": {
                "model_version": "v2_new_schema",
                "note": "orientation_sexuelle exclue pour raisons éthiques"
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
