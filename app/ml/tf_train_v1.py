import os
from datetime import datetime

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf

from ..database import SessionLocal
from .. import models


def load_data_old_schema():
    """
    Charge les données depuis la base en utilisant uniquement
    les features du 'vieux' schéma (sans nb_enfants, quotient_caf).

    Retourne :
    - X : features prêtes pour le modèle (numpy array)
    - y : cible (score_credit)
    - encoder : OneHotEncoder pour les variables catégorielles
    - scaler : StandardScaler pour les features concaténées
    """
    db: Session = SessionLocal()
    try:
        clients = (
            db.query(models.Client)
            .join(models.FinancialInfo)
            .all()
        )

        numeric_features = []
        categorical_features = []
        y = []

        for client in clients:
            fin = client.financial_info
            if fin is None or fin.credit_score is None:
                continue

            y.append(fin.credit_score)

            numeric_features.append([
                client.age,
                client.height_cm or 0.0,
                client.weight_kg or 0.0,
                fin.monthly_income or 0.0,
                fin.credit_history or 0.0,
                fin.personal_risk or 0.0,
                fin.monthly_rent or 0.0,
                fin.loan_amount or 0.0,
            ])

            categorical_features.append([
                client.sex,
                client.sport_licence,
                client.education_level,
                client.region,
                client.smoker,
                client.is_french,
                client.family_status,
            ])

        if not numeric_features:
            raise ValueError("Aucune donnée exploitable trouvée en base.")

        numeric_features = np.array(numeric_features, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        cat_encoded = encoder.fit_transform(categorical_features)

        X_raw = np.concatenate([numeric_features, cat_encoded], axis=1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        return X_scaled, y, encoder, scaler

    finally:
        db.close()


def build_model_v1(input_dim: int) -> tf.keras.Model:
    """
    Construit le modèle Keras 'v1' (ancien schéma).
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="inputs")
    x = tf.keras.layers.Dense(32, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dense(16, activation="relu", name="dense_2")(x)
    outputs = tf.keras.layers.Dense(1, name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="credit_score_v1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mse"],
    )
    return model


def train_and_save_v1():
    # 1) Data
    X, y, encoder, scaler = load_data_old_schema()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_dim = X.shape[1]
    print(f"Dimensions des features (ancien schéma) : {input_dim}")

    # 2) Modèle
    model = build_model_v1(input_dim=input_dim)

    # 3) Callbacks TensorBoard + EarlyStopping
    os.makedirs("logs/v1", exist_ok=True)
    log_dir = os.path.join("logs", "v1", datetime.now().strftime("%Y%m%d-%H%M%S"))

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    # 4) Entraînement
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[tensorboard_cb, earlystop_cb],
        verbose=1,
    )

    # 5) Évaluation
    y_val_pred = model.predict(X_val)
    mse_val = mean_squared_error(y_val, y_val_pred)
    print(f"[V1] MSE validation finale : {mse_val:.4f}")

    # 6) Sauvegardes
    os.makedirs("artifacts", exist_ok=True)

    model_path = os.path.join("artifacts", "model_v1_old_schema.h5")
    model.save(model_path)
    print(f"Modèle v1 sauvegardé dans {model_path}")

    encoder_path = os.path.join("artifacts", "encoder_v1.pkl")
    scaler_path = os.path.join("artifacts", "scaler_v1.pkl")

    joblib.dump(encoder, encoder_path)
    joblib.dump(scaler, scaler_path)

    print(f"Encoder sauvegardé dans {encoder_path}")
    print(f"Scaler sauvegardé dans {scaler_path}")

    # 7) Courbe de loss
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure()
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training & Validation Loss - Model V1 (old schema)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("artifacts", "loss_curve_v1.png"))
    plt.close()
    print("Courbe de loss v1 sauvegardée dans artifacts/loss_curve_v1.png")

    print(f"Logs TensorBoard disponibles dans : {log_dir}")
    print("Lancez : tensorboard --logdir logs pour visualiser.")


if __name__ == "__main__":
    train_and_save_v1()
