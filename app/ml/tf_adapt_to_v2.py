import os
import numpy as np
import tensorflow as tf


def adapt_model_v1_to_v2(
    model_v1_path: str = "artifacts/model_v1_old_schema.h5",
    model_v2_path: str = "artifacts/model_v2_new_schema_init.h5",
    n_new_features: int = 2,
):
    """
    Adapte le modèle v1 (ancien schéma de features) en un modèle v2
    avec une couche d'entrée élargie (pour plus de colonnes).

    - On charge le modèle v1
    - On construit un nouveau modèle v2 avec :
        * nouvelle Input(shape=(old_input_dim + n_new_features,))
        * même structure interne (Dense 32 -> Dense 16 -> Dense 1)
    - On recopie les poids :
        * première Dense : on étend les poids d'entrée
        * couches suivantes : copiées à l'identique
    - On sauvegarde le modèle v2 initial dans model_v2_path.
    """

    if not os.path.exists(model_v1_path):
        raise FileNotFoundError(f"Modèle v1 introuvable : {model_v1_path}")

    print(f"Chargement du modèle v1 depuis : {model_v1_path}")
    model_v1 = tf.keras.models.load_model(model_v1_path, compile=False)

    # Récupération de la dimension d'entrée "ancienne"
    old_input_dim = model_v1.input_shape[-1]
    print(f"Ancienne dimension d'entrée : {old_input_dim}")

    # On suppose la première couche cachée s'appelle "dense_1"
    dense1_v1 = model_v1.get_layer("dense_1")
    W_old, b_old = dense1_v1.get_weights()
    old_input_dim_W, units = W_old.shape

    assert old_input_dim_W == old_input_dim, (
        f"Incohérence dimensions : input_shape={old_input_dim}, "
        f"W_old.shape[0]={old_input_dim_W}"
    )

    print(f"Première couche Dense (dense_1) : input_dim={old_input_dim_W}, units={units}")

    # Nouvelle dimension d'entrée (on ajoute n_new_features colonnes)
    new_input_dim = old_input_dim + n_new_features
    print(f"Nouvelle dimension d'entrée : {new_input_dim}")

    # Construction du nouveau modèle v2
    inputs_v2 = tf.keras.Input(shape=(new_input_dim,), name="inputs_v2")
    x = tf.keras.layers.Dense(units, activation="relu", name="dense_1_v2")(inputs_v2)
    x = tf.keras.layers.Dense(16, activation="relu", name="dense_2_v2")(x)
    outputs_v2 = tf.keras.layers.Dense(1, name="output_v2")(x)

    model_v2 = tf.keras.Model(inputs=inputs_v2, outputs=outputs_v2, name="credit_score_v2")

    # Copie / adaptation des poids

    # 1) Première Dense : on étend la matrice W_old au nouveau input_dim
    #    - les anciennes features gardent leurs poids
    #    - les nouvelles features ont des poids initiaux faibles (par ex. ~N(0, 0.01))
    rng = np.random.default_rng(42)
    W_new = rng.normal(0, 0.01, size=(new_input_dim, units))
    W_new[:old_input_dim, :] = W_old

    dense1_v2 = model_v2.get_layer("dense_1_v2")
    dense1_v2.set_weights([W_new, b_old])
    print("Poids de dense_1 adaptés et copiés vers dense_1_v2.")

    # 2) Deuxième Dense : même shape entre v1 et v2 → copie directe
    dense2_v1 = model_v1.get_layer("dense_2")
    dense2_v2 = model_v2.get_layer("dense_2_v2")
    dense2_v2.set_weights(dense2_v1.get_weights())
    print("Poids de dense_2 copiés vers dense_2_v2.")

    # 3) Couche de sortie : même shape → copie directe
    output_v1 = model_v1.get_layer("output")
    output_v2 = model_v2.get_layer("output_v2")
    output_v2.set_weights(output_v1.get_weights())
    print("Poids de output copiés vers output_v2.")

    # Compilation du modèle v2 (même config)
    model_v2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mse"],
    )

    # Sauvegarde
    os.makedirs("artifacts", exist_ok=True)
    model_v2.save(model_v2_path)
    print(f"Modèle v2 initial sauvegardé dans : {model_v2_path}")


if __name__ == "__main__":
    adapt_model_v1_to_v2()
