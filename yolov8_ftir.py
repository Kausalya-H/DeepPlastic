import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ==================================================
# 1. REPRODUCIBILITY
# ==================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==================================================
# 2. LOAD DATASET
# ==================================================

DATA_PATH = (
    "FTIR-PLASTIC/FTIR-PLASTIC-c8/"
    "FTIR_PLASTIC_c8.csv"
)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nSamples per polymer:")
print(df["Polymer"].value_counts())


# ==================================================
# 3. EXTRACT SPECTRAL FEATURES
# ==================================================

spectral_columns = [
    col for col in df.columns
    if col.startswith("Data(Y)")
]

if not spectral_columns:
    raise ValueError("No Data(Y) spectral columns found.")

X = df[spectral_columns].apply(
    pd.to_numeric,
    errors="coerce"
).to_numpy(dtype=np.float32, copy=True)

y_text = df["Polymer"].astype(str).to_numpy()


# ==================================================
# 4. CHECK DUPLICATE SPECTRA
# ==================================================

print("\n========== DUPLICATE CHECK ==========")

spectra_df = pd.DataFrame(X)

print(
    "Exact duplicate spectra:",
    int(spectra_df.duplicated().sum())
)

# Check whether identical spectra have conflicting labels.
spectra_with_labels = spectra_df.copy()
spectra_with_labels["Polymer"] = y_text

duplicate_label_groups = (
    spectra_with_labels
    .groupby(
        list(spectra_df.columns),
        dropna=False
    )["Polymer"]
    .nunique()
)

print(
    "Identical spectra with different labels:",
    int((duplicate_label_groups > 1).sum())
)

# Display possible specimen identifiers for inspection.
available_columns = [
    col for col in ["IDE", "Polymer", "Sample", "Technic"]
    if col in df.columns
]

if available_columns:
    print("\nSample information:")
    print(df[available_columns].head(15))


# ==================================================
# 5. HANDLE MISSING VALUES
# ==================================================

column_medians = np.nanmedian(X, axis=0)

column_medians = np.nan_to_num(
    column_medians,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

bad_rows, bad_cols = np.where(~np.isfinite(X))

X[bad_rows, bad_cols] = column_medians[bad_cols]


# ==================================================
# 6. ENCODE LABELS
# ==================================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

NUM_CLASSES = len(label_encoder.classes_)

print("\nPolymer classes:", list(label_encoder.classes_))
print("Number of spectral features:", X.shape[1])


# ==================================================
# 7. SPLIT DATA
# ==================================================

# Keep the test set untouched during model development.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=SEED,
    stratify=y_train
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


# ==================================================
# 8. STANDARDIZE SPECTRAL FEATURES
# ==================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = np.clip(X_train, -5, 5)
X_val = np.clip(X_val, -5, 5)
X_test = np.clip(X_test, -5, 5)


# ==================================================
# 9. PREPARE 1D SPECTRAL INPUTS
# ==================================================

# The Transformer processes the FTIR spectrum directly.
# Each spectral position is represented as one token.

NUM_FEATURES = X_train.shape[1]

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("\n1D Transformer input shape:", X_train.shape)


# ==================================================
# 10. TRANSFORMER CONFIGURATION
# ==================================================

PATCH_SIZE = 20
EMBED_DIM = 16
NUM_HEADS = 2
MLP_DIM = 32
NUM_TRANSFORMER_BLOCKS = 1

DROPOUT_RATE = 0.45
L2_STRENGTH = 0.003


# ==================================================
# 11. PATCH EMBEDDING
# ==================================================

@tf.keras.utils.register_keras_serializable()
class AddPositionEmbedding(layers.Layer):

    def build(self, input_shape):
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(1, input_shape[1], input_shape[2]),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):
        return inputs + self.position_embedding


def transformer_block(
    x,
    embed_dim,
    num_heads,
    mlp_dim,
    dropout_rate
):

    # Self-attention.
    shortcut = x

    x = layers.LayerNormalization(
        epsilon=1e-6
    )(x)

    attention = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        dropout=dropout_rate
    )(x, x)

    attention = layers.Dropout(dropout_rate)(attention)

    x = layers.Add()([shortcut, attention])

    # Feed-forward network.
    shortcut = x

    x = layers.LayerNormalization(
        epsilon=1e-6
    )(x)

    x = layers.Dense(
        mlp_dim,
        activation="gelu",
        kernel_regularizer=regularizers.l2(L2_STRENGTH)
    )(x)

    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(
        embed_dim,
        kernel_regularizer=regularizers.l2(L2_STRENGTH)
    )(x)

    x = layers.Dropout(dropout_rate)(x)

    return layers.Add()([shortcut, x])


# ==================================================
# 12. BUILD 1D TRANSFORMER
# ==================================================

inputs = layers.Input(
    shape=(NUM_FEATURES, 1)
)

# Small random perturbations during training.
x = layers.GaussianNoise(0.05)(inputs)

# Group neighboring spectral values into patches.
x = layers.Conv1D(
    filters=EMBED_DIM,
    kernel_size=PATCH_SIZE,
    strides=PATCH_SIZE,
    padding="valid",
    kernel_regularizer=regularizers.l2(L2_STRENGTH)
)(x)

# Learnable position information.
x = AddPositionEmbedding()(x)

x = layers.Dropout(DROPOUT_RATE)(x)

# Transformer encoder.
for _ in range(NUM_TRANSFORMER_BLOCKS):
    x = transformer_block(
        x,
        EMBED_DIM,
        NUM_HEADS,
        MLP_DIM,
        DROPOUT_RATE
    )

# Classification head.
x = layers.LayerNormalization(epsilon=1e-6)(x)

x = layers.GlobalAveragePooling1D()(x)

x = layers.Dense(
    16,
    activation="gelu",
    kernel_regularizer=regularizers.l2(L2_STRENGTH)
)(x)

x = layers.Dropout(0.50)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="FTIR_1D_Transformer"
)


# ==================================================
# 13. COMPILE MODEL
# ==================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==================================================
# 14. TRAIN MODEL
# ==================================================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),

    ModelCheckpoint(
        "best_ftir_1d_transformer.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)


# ==================================================
# 15. TEST EVALUATION
# ==================================================

probabilities = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n========== 1D TRANSFORMER RESULTS ==========")

print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ==================================================
# 16. CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("FTIR Plastic Classification - 1D Transformer")
plt.tight_layout()
plt.savefig("ftir_1d_transformer_confusion_matrix.png", dpi=300)
plt.show()


# ==================================================
# 17. ACCURACY GRAPH
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("1D Transformer Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig("ftir_1d_transformer_accuracy.png", dpi=300)
plt.show()


# ==================================================
# 18. LOSS GRAPH
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("1D Transformer Loss")
plt.legend()
plt.tight_layout()

plt.savefig("ftir_1d_transformer_loss.png", dpi=300)
plt.show()


# ==================================================
# 19. SAVE MODEL
# ==================================================

model.save("final_ftir_1d_transformer.keras")

print("\nTraining completed.")
print("Best model: best_ftir_1d_transformer.keras")
print("Final model: final_ftir_1d_transformer.keras")
print("Confusion matrix: ftir_1d_transformer_confusion_matrix.png")
print("Accuracy graph: ftir_1d_transformer_accuracy.png")
print("Loss graph: ftir_1d_transformer_loss.png")