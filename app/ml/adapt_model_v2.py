import os
import numpy as np
import tensorflow as tf


def load_v1_model(model_path: str) -> tf.keras.Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle v1 introuvable à l'emplacement : {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Modèle v1 chargé depuis {model_path}")
    model.summary()
    return model


def build_v2_from_v1(model_v1: tf.keras.Model, extra_input_dim: int = 2) -> tf.keras.Model:
    """
    Construit un modèle v2 en :
    - augmentant la dimension d'entrée de `extra_input_dim`
    - recopiant les poids internes de v1

    On suppose l'architecture suivante pour v1 :
    Input -> Dense(32, relu, name='dense_1')
           -> Dense(16, relu, name='dense_2')
           -> Dense(1, name='output')
    """
    # Récupérer la première couche dense du modèle v1
    dense1_old = model_v1.get_layer("dense_1")
    W_old, b_old = dense1_old.get_weights()  # W_old: (old_input_dim, units1)

    old_input_dim, units1 = W_old.shape
    print(f"Ancienne dimension d'entrée : {old_input_dim}")
    print(f"Nombre de neurones de dense_1 : {units1}")

    # Nouvelle dimension d'entrée = ancienne + nb de nouvelles features
    new_input_dim = old_input_dim + extra_input_dim
    print(f"Nouvelle dimension d'entrée (v2) : {new_input_dim}")

    # Construire la nouvelle architecture v2
    inputs = tf.keras.Input(shape=(new_input_dim,), name="inputs")
    x = tf.keras.layers.Dense(units1, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dense(
        model_v1.get_layer("dense_2").units,
        activation="relu",
        name="dense_2",
    )(x)
    outputs = tf.keras.layers.Dense(1, name="output")(x)

    model_v2 = tf.keras.Model(inputs=inputs, outputs=outputs, name="credit_score_v2")

    # --- Copier les poids ---

    # 1) Première couche dense : on étend les poids
    # W_old: (old_input_dim, units1)
    # W_new: (new_input_dim, units1)
    W_new = np.zeros((new_input_dim, units1), dtype=W_old.dtype)
    # On recopie les poids pour les anciennes features
    W_new[:old_input_dim, :] = W_old
    # Les 2 nouvelles features (nb_enfants, quotient_caf) ont au départ des poids nuls

    dense1_v2 = model_v2.get_layer("dense_1")
    dense1_v2.set_weights([W_new, b_old])
    print("Poids de dense_1 recopiés et étendus pour les nouvelles entrées.")

    # 2) Autres couches : copie directe
    for layer_name in ["dense_2", "output"]:
        layer_v1 = model_v1.get_layer(layer_name)
        layer_v2 = model_v2.get_layer(layer_name)
        layer_v2.set_weights(layer_v1.get_weights())
        print(f"Poids de la couche {layer_name} recopiés.")

    # Compilation du modèle v2 (mêmes hyperparamètres que v1)
    model_v2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mse"],
    )

    print("Modèle v2 construit et compilé.")
    model_v2.summary()

    return model_v2


def main():
    os.makedirs("artifacts", exist_ok=True)

    # 1) Charger le modèle v1
    model_v1_path = os.path.join("artifacts", "model_v1_old_schema.h5")
    model_v1 = load_v1_model(model_v1_path)

    # 2) Construire v2 avec 2 nouvelles features (nb_enfants, quotient_caf)
    model_v2 = build_v2_from_v1(model_v1, extra_input_dim=2)

    # 3) Sauvegarder le modèle v2 initialisé (avant réentraînement)
    model_v2_path = os.path.join("artifacts", "model_v2_new_schema_init.h5")
    model_v2.save(model_v2_path)
    print(f"Modèle v2 (initialisé) sauvegardé dans {model_v2_path}")


if __name__ == "__main__":
    main()
