import os
from datetime import datetime

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf

from ..database import SessionLocal
from .. import models


def load_data_new_schema(encoder_v1, scaler_v1):
    """
    Charge les données depuis la base en utilisant :
    - les mêmes features que v1 (anciennes features)
    - + 2 nouvelles features : nb_enfants, quotient_caf

    encoder_v1 : OneHotEncoder déjà fit sur les anciennes colonnes catégorielles
    scaler_v1 : StandardScaler déjà fit sur les anciennes features concaténées

    Retourne :
    - X_full : features complètes (anciennes + nouvelles), déjà scalées
    - y : cible (score_credit)
    - extra_scaler : scaler utilisé uniquement pour [nb_enfants, quotient_caf]
    """
    db: Session = SessionLocal()
    try:
        clients = (
            db.query(models.Client)
            .join(models.FinancialInfo)
            .all()
        )

        numeric_old = []
        categorical = []
        extra_numeric = []  # [nb_enfants, quotient_caf]
        y = []

        for client in clients:
            fin = client.financial_info
            if fin is None or fin.credit_score is None:
                continue

            y.append(fin.credit_score)

            # Numériques "anciennes" (comme v1)
            numeric_old.append([
                client.age,
                client.height_cm or 0.0,
                client.weight_kg or 0.0,
                fin.monthly_income or 0.0,
                fin.credit_history or 0.0,
                fin.personal_risk or 0.0,
                fin.monthly_rent or 0.0,
                fin.loan_amount or 0.0,
            ])

            # Catégorielles (comme v1)
            categorical.append([
                client.sex,
                client.sport_licence,
                client.education_level,
                client.region,
                client.smoker,
                client.is_french,
                client.family_status,
            ])

            # Nouvelles features numériques
            extra_numeric.append([
                client.nb_enfants or 0,
                client.quotient_caf or 0.0,
            ])

        if not numeric_old:
            raise ValueError("Aucune donnée exploitable trouvée en base.")

        numeric_old = np.array(numeric_old, dtype=np.float32)
        extra_numeric = np.array(extra_numeric, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        # Encodage catégoriel avec l'encoder v1 (on ne refit pas)
        cat_encoded = encoder_v1.transform(categorical)

        # Anciennes features combinées
        X_old = np.concatenate([numeric_old, cat_encoded], axis=1)

        # Scalées avec scaler_v1 → cohérence avec les poids internes du modèle v1
        X_old_scaled = scaler_v1.transform(X_old)

        # Nouveau scaler uniquement pour [nb_enfants, quotient_caf]
        extra_scaler = StandardScaler()
        extra_scaled = extra_scaler.fit_transform(extra_numeric)

        # Features complètes pour v2 : [ancien_schéma_scalé || nouvelles_features_scalées]
        X_full = np.concatenate([X_old_scaled, extra_scaled], axis=1)

        return X_full, y, extra_scaler

    finally:
        db.close()


def train_and_save_v2():
    # 1) Charger encoder & scaler de v1
    encoder_v1_path = os.path.join("artifacts", "encoder_v1.pkl")
    scaler_v1_path = os.path.join("artifacts", "scaler_v1.pkl")

    if not os.path.exists(encoder_v1_path) or not os.path.exists(scaler_v1_path):
        raise FileNotFoundError("encoder_v1.pkl ou scaler_v1.pkl introuvable. "
                                "Assurez-vous d'avoir exécuté tf_train_v1 d'abord.")

    encoder_v1 = joblib.load(encoder_v1_path)
    scaler_v1 = joblib.load(scaler_v1_path)

    # 2) Charger les données avec le nouveau schéma
    X, y, extra_scaler = load_data_new_schema(encoder_v1, scaler_v1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_dim_v2 = X.shape[1]
    print(f"Dimensions des features (nouveau schéma) : {input_dim_v2}")

    # 3) Charger le modèle v2 "adapté" (couche d'entrée élargie + poids internes v1)
    model_v2_init_path = os.path.join("artifacts", "model_v2_new_schema_init.h5")
    if not os.path.exists(model_v2_init_path):
        raise FileNotFoundError("model_v2_new_schema_init.h5 introuvable. "
                                "Assurez-vous d'avoir exécuté tf_adapt_to_v2 d'abord.")

    # On le charge sans compile, puis on recompile (Keras 3)
    model_v2 = tf.keras.models.load_model(model_v2_init_path, compile=False)
    model_v2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mse"],
    )

    # 4) Callbacks TensorBoard + EarlyStopping
    os.makedirs("logs/v2", exist_ok=True)
    log_dir = os.path.join("logs", "v2", datetime.now().strftime("%Y%m%d-%H%M%S"))

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    # 5) Entraînement
    history = model_v2.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[tensorboard_cb, earlystop_cb],
        verbose=1,
    )

    # 6) Évaluation
    y_val_pred = model_v2.predict(X_val)
    mse_val = mean_squared_error(y_val, y_val_pred)
    print(f"[V2] MSE validation finale : {mse_val:.4f}")

    # 7) Sauvegardes
    os.makedirs("artifacts", exist_ok=True)

    model_v2_final_path = os.path.join("artifacts", "model_v2_new_schema.h5")
    model_v2.save(model_v2_final_path)
    print(f"Modèle v2 final sauvegardé dans {model_v2_final_path}")

    extra_scaler_path = os.path.join("artifacts", "extra_scaler_v2.pkl")
    joblib.dump(extra_scaler, extra_scaler_path)
    print(f"Scaler des nouvelles features sauvegardé dans {extra_scaler_path}")

    # 8) Courbe de loss
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure()
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training & Validation Loss - Model V2 (new schema)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("artifacts", "loss_curve_v2.png"))
    plt.close()
    print("Courbe de loss v2 sauvegardée dans artifacts/loss_curve_v2.png")

    print(f"Logs TensorBoard V2 disponibles dans : {log_dir}")
    print("Lancez : tensorboard --logdir logs pour visualiser.")


if __name__ == "__main__":
    train_and_save_v2()
