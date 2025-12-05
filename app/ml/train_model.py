import os
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

from ..database import SessionLocal
from .. import models


def load_data_from_db():
    """
    Charge les données depuis la base :
    - Features numériques + catégorielles (INCLUANT nb_enfants et quotient_caf)
    - Target : credit_score

    ⚠ NOTE ETHIQUE :
    La colonne 'orientation_sexuelle' est VOLONTAIREMENT
    exclue des features pour éviter l'introduction de biais
    discriminatoires liés à ce type de donnée sensible.
    """
    db: Session = SessionLocal()
    try:
        clients = db.query(models.Client).join(models.FinancialInfo).all()

        if not clients:
            raise ValueError("Aucun client avec FinancialInfo en base.")

        numeric_features = []
        categorical_features = []
        y = []

        for client in clients:
            fin = client.financial_info
            # On ne garde une ligne que si on a un score de crédit
            if fin is None or fin.credit_score is None:
                continue

            # Target
            y.append(fin.credit_score)

            # Variables numériques
            numeric_features.append([
                client.age,
                client.height_cm or 0.0,
                client.weight_kg or 0.0,
                client.nb_enfants or 0,          # NOUVEAU
                client.quotient_caf or 0.0,      # NOUVEAU
                fin.monthly_income or 0.0,
                fin.credit_history or 0.0,
                fin.personal_risk or 0.0,
                fin.monthly_rent or 0.0,
                fin.loan_amount or 0.0,
            ])

            # Variables catégorielles
            categorical_features.append([
                client.sex,
                client.sport_licence,
                client.education_level,
                client.region,
                client.smoker,
                client.is_french,
                client.family_status,
                # ⚠ PAS d'orientation_sexuelle ici volontairement
            ])

        numeric_features = np.array(numeric_features, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        # Encodage One-Hot des variables catégorielles
        encoder = OneHotEncoder(sparse_output=False)
        cat_encoded = encoder.fit_transform(categorical_features)

        # Concaténation numérique + catégoriel
        X = np.concatenate([numeric_features, cat_encoded], axis=1)

        return X, y
    finally:
        db.close()


class SimpleNN:
    """
    Petit réseau de neurones fully-connected :
    input_dim -> 32 -> 16 -> 1
    ReLU + MSE, entraîné par gradient descent en NumPy
    """

    def __init__(self, input_dim, hidden1=32, hidden2=16, lr=1e-3):
        rng = np.random.default_rng(42)

        self.W1 = rng.normal(0, 0.1, size=(input_dim, hidden1))
        self.b1 = np.zeros((1, hidden1))

        self.W2 = rng.normal(0, 0.1, size=(hidden1, hidden2))
        self.b2 = np.zeros((1, hidden2))

        self.W3 = rng.normal(0, 0.1, size=(hidden2, 1))
        self.b3 = np.zeros((1, 1))

        self.lr = lr

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_deriv(x):
        return (x > 0).astype(np.float32)

    def forward(self, X):
        """
        Renvoie les sorties intermédiaires pour backprop
        """
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)

        z3 = a2 @ self.W3 + self.b3
        y_pred = z3  # régression linéaire

        cache = (X, z1, a1, z2, a2, z3, y_pred)
        return y_pred, cache

    def compute_loss(self, y_pred, y_true):
        """
        MSE
        """
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, cache, y_true):
        X, z1, a1, z2, a2, z3, y_pred = cache
        m = y_true.shape[0]

        # dL/dy_pred
        dL_dy = 2 * (y_pred - y_true) / m  # (m, 1)

        # Layer 3
        dL_dW3 = a2.T @ dL_dy          # (hidden2, 1)
        dL_db3 = np.sum(dL_dy, axis=0, keepdims=True)

        # Backprop into layer 2
        dL_da2 = dL_dy @ self.W3.T      # (m, hidden2)
        dL_dz2 = dL_da2 * self.relu_deriv(z2)

        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        # Backprop into layer 1
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * self.relu_deriv(z1)

        dL_dW1 = X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        # Gradient descent step
        self.W3 -= self.lr * dL_dW3
        self.b3 -= self.lr * dL_db3

        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2

        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        n_samples = X_train.shape[0]
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # mini-batch
            indices = np.random.permutation(n_samples)
            X_train_shuff = X_train[indices]
            y_train_shuff = y_train[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                xb = X_train_shuff[start:end]
                yb = y_train_shuff[start:end]

                y_pred, cache = self.forward(xb)
                self.backward(cache, yb)

            # compute losses at epoch end
            y_train_pred, _ = self.forward(X_train)
            y_val_pred, _ = self.forward(X_val)

            train_loss = self.compute_loss(y_train_pred, y_train)
            val_loss = self.compute_loss(y_val_pred, y_val)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        return train_losses, val_losses

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return y_pred


def train_and_save():
    X, y = load_data_from_db()

    # Standardisation des features pour stabiliser l'entraînement
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = SimpleNN(input_dim=X_scaled.shape[1], hidden1=32, hidden2=16, lr=1e-3)

    train_losses, val_losses = model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32,
    )

    # Évaluation finale
    y_val_pred = model.predict(X_val)
    mse_val = mean_squared_error(y_val, y_val_pred)
    print(f"MSE validation finale : {mse_val:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    # Sauvegarde des poids
    np.savez(
        "artifacts/credit_score_model_weights.npz",
        W1=model.W1, b1=model.b1,
        W2=model.W2, b2=model.b2,
        W3=model.W3, b3=model.b3,
    )

    # Courbe de loss
    plt.figure()
    plt.plot(train_losses, label="train_loss", linestyle="--")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.title("Training & Validation Loss - Credit Score Model (NumPy NN)")
    plt.grid(True)
    plt.savefig("artifacts/loss_curve.png")
    plt.close()

    print("Poids du modèle sauvegardés dans artifacts/credit_score_model_weights.npz")
    print("Courbe de loss sauvegardée dans artifacts/loss_curve.png")


if __name__ == "__main__":
    train_and_save()
