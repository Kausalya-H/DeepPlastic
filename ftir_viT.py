import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
)
warnings.filterwarnings("ignore")
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
DATA_PATH = (
    "FTIR-PLASTIC/FTIR-PLASTIC-c8/"
    "FTIR_PLASTIC_c8.csv"
)
MODEL_PATH = "best_vit_ftir.keras"
PATCH_SIZE = 32
EMBED_DIM = 24
NUM_HEADS = 2
MLP_DIM = 32
NUM_BLOCKS = 1
DROPOUT_RATE = 0.35
CLASSIFIER_DROPOUT = 0.40
L2_STRENGTH = 0.001
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
EPOCHS = 60
TARGET_CANDIDATES = ["Polymer", "polymer", "POLYMER"]
METADATA_COLUMNS = ["IDE","ID","Polymer","polymer","POLYMER","Technic","Technique","technics",]
@tf.keras.utils.register_keras_serializable(package="FTIRViT")
class SpectralPatchEmbedding(layers.Layer):
    def __init__(self,patch_size=32,embed_dim=24,**kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = layers.Conv1D(filters=embed_dim,kernel_size=patch_size,strides=patch_size,padding="valid",kernel_regularizer=regularizers.l2(L2_STRENGTH),)
        
    def call(self, inputs):
        input_length = tf.shape(inputs)[1]
        remainder = tf.math.mod(input_length,self.patch_size)
        padding_length = tf.math.mod(self.patch_size - remainder,self.patch_size)
        inputs = tf.pad(
            inputs,
            paddings=[[0, 0],[0, padding_length],[0, 0],],)
        return self.projection(inputs)
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size,"embed_dim": self.embed_dim,})
        return config
@tf.keras.utils.register_keras_serializable(package="FTIRViT")

class SpectralPositionEmbedding(layers.Layer):

    def __init__(self,regularization_strength=0.001,**kwargs):
        super().__init__(**kwargs)
        self.regularization_strength = (regularization_strength)

    def build(self, input_shape):
        sequence_length = input_shape[1]
        embedding_dimension = input_shape[2]
        if sequence_length is None:
            raise ValueError(
                "The patch sequence length must be known."
            )
        self.position_embedding = self.add_weight(name="position_embedding",shape=(1,sequence_length,embedding_dimension,),initializer="random_normal",trainable=True,regularizer=regularizers.l2(self.regularization_strength),)
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.position_embedding
    
    def get_config(self):
        config = super().get_config()
        config.update({"regularization_strength":self.regularization_strength,})
        return config

def transformer_encoder(inputs,embed_dim,num_heads,mlp_dim,dropout_rate,l2_strength,):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = layers.MultiHeadAttention(num_heads=num_heads,key_dim=embed_dim // num_heads,dropout=dropout_rate,)(x,x,)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    x = layers.Add()([inputs,attention_output,])
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(mlp_dim,activation="gelu",kernel_regularizer=regularizers.l2(l2_strength),)(y)
    y = layers.Dropout(dropout_rate)(y)
    y = layers.Dense(embed_dim,kernel_regularizer=regularizers.l2(l2_strength),)(y)
    y = layers.Dropout(dropout_rate)(y)
    return layers.Add()([x, y])

def build_vit(number_of_features, number_of_classes):
    inputs = layers.Input(shape=(number_of_features, 1),name="spectral_input",)
    x = SpectralPatchEmbedding(patch_size=PATCH_SIZE,embed_dim=EMBED_DIM,name="spectral_patch_embedding",)(inputs)
    x = SpectralPositionEmbedding(regularization_strength=L2_STRENGTH,name="spectral_position_embedding",)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    for block_number in range(NUM_BLOCKS):
        x = transformer_encoder(inputs=x,embed_dim=EMBED_DIM,num_heads=NUM_HEADS,mlp_dim=MLP_DIM,dropout_rate=DROPOUT_RATE,l2_strength=L2_STRENGTH,)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(32,activation="gelu",kernel_regularizer=regularizers.l2(L2_STRENGTH),)(x)
    x = layers.Dropout(CLASSIFIER_DROPOUT)(x)
    outputs = layers.Dense(number_of_classes,activation="softmax",name="polymer_prediction",)(x)
    model = Model(inputs=inputs,outputs=outputs,name="FTIR_Vision_Transformer",)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE,weight_decay=0.0001,clipnorm=1.0,),loss="sparse_categorical_crossentropy",metrics=["accuracy"],)
    return model
print("\n" + "=" * 65)
print("FTIR PLASTIC CLASSIFICATION - VISION TRANSFORMER")
print("=" * 65)
if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}\n"
        "Run this script from your FTIR(DL) project folder."
    )
df = pd.read_csv(DATA_PATH)
print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)
print("\nColumn names (first 15):")
print(df.columns[:15].tolist())
target_column = next(
    (
        column
        for column in TARGET_CANDIDATES
        if column in df.columns
    ),
    None,
)

if target_column is None:

    raise ValueError(
        "Could not find the Polymer target column. "
        f"Available columns include: {df.columns[:30].tolist()}"
    )
df = df.dropna(subset=[target_column]).reset_index(drop=True)
y_text = df[target_column].astype(str).str.strip()
valid_rows = y_text.ne("")
df = df.loc[valid_rows].reset_index(drop=True)
y_text = y_text.loc[valid_rows].reset_index(drop=True)
columns_to_exclude = set(METADATA_COLUMNS)
columns_to_exclude.add(target_column)
feature_df = df.drop(
    columns=[column for column in df.columns if column in columns_to_exclude],errors="ignore",)
feature_df = feature_df.select_dtypes(
    include=[np.number]
)
if feature_df.shape[1] == 0:
    raise ValueError("No numeric spectral features were found.")
feature_df = feature_df.replace(
    [np.inf, -np.inf],
    np.nan,
)
feature_df = feature_df.dropna(axis=1,how="all",)
if feature_df.shape[1] == 0:
    raise ValueError(
        "All numerical features are empty."
    )
feature_df = feature_df.fillna(
    feature_df.median()
)
non_constant_columns = (feature_df.nunique(dropna=False) > 1)
feature_df = feature_df.loc[:,non_constant_columns]
if feature_df.shape[1] == 0:
    raise ValueError(
        "No non-constant numeric spectral features remain."
    )
X = feature_df.to_numpy(
    dtype=np.float32
)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(
    y_text.to_numpy()
)
class_names = label_encoder.classes_
print("\nNumber of usable samples:", len(X))
print("Number of spectral features:", X.shape[1])
print("Number of polymer classes:", len(class_names))
print("\nClass distribution:")
print(pd.Series(y_text.to_numpy()).value_counts().sort_index())
print("\nClass encoding:")
for index, class_name in enumerate(class_names):
    print(f"{index} = {class_name}")
if len(class_names) < 2:
    raise ValueError(
        "At least two polymer classes are required."
    )
print("\nFirst 10 input feature names:")
print(feature_df.columns[:10].tolist())
print("\nChecking for exact duplicate spectra...")
spectra_hash = pd.util.hash_pandas_object(feature_df,index=False,).to_numpy()
hash_table = pd.DataFrame({
    "hash": spectra_hash,
    "label": y,
})
duplicate_groups = hash_table.groupby(
    "hash"
)["label"].nunique()

conflicting_duplicate_groups = (
    duplicate_groups > 1
).sum()

print(
    "Number of distinct exact-duplicate spectra:",
    int((hash_table["hash"].duplicated(keep=False)).sum()),
)

print(
    "Duplicate spectral groups with conflicting labels:",
    int(conflicting_duplicate_groups),
)

if conflicting_duplicate_groups > 0:

    print(
        "WARNING: Identical spectra have different labels. "
        "Investigate these records before interpreting results."
    )
X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.40,random_state=SEED,stratify=y,)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.50,random_state=SEED,stratify=y_temp,)
print("\nData split:")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

X_train = np.nan_to_num(X_train,nan=0.0,posinf=0.0,neginf=0.0,)
X_val = np.nan_to_num(X_val,nan=0.0,posinf=0.0,neginf=0.0,)
X_test = np.nan_to_num(X_test,nan=0.0,posinf=0.0,neginf=0.0,)
print("\nInput shape:", X_train.shape)
model = build_vit(number_of_features=X_train.shape[1],number_of_classes=len(class_names),)
print("\nModel architecture:")
model.summary()
callbacks = [
    ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=6,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),

    CSVLogger(
        "vit_training_log.csv",
        append=False,
    ),
]

# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\nStarting training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True,
    verbose=1,
)
print("\nTraining completed.")
if os.path.exists(MODEL_PATH):

    print("\nLoading best saved model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )
def evaluate_split(
    model,
    X_data,
    y_true,
    split_name,
):
    probabilities = model.predict(
        X_data,
        verbose=0,
    )
    y_pred = np.argmax(
        probabilities,
        axis=1,
    )
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )
    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    print("\n" + "=" * 65)
    print(f"{split_name.upper()} RESULTS")
    print("=" * 65)
    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )
    print("Confusion matrix:")
    print(cm)
    return {
        "split": split_name,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "predictions": y_pred,
        "probabilities": probabilities,
        "confusion_matrix": cm,
    }
train_results = evaluate_split(
    model,
    X_train,
    y_train,
    "Training",
)
val_results = evaluate_split(
    model,
    X_val,
    y_val,
    "Validation",
)
test_results = evaluate_split(
    model,
    X_test,
    y_test,
    "Test",
)
train_accuracy = train_results["accuracy"]
val_accuracy = val_results["accuracy"]
test_accuracy = test_results["accuracy"]
print("\n" + "=" * 65)
print("GENERALIZATION CHECK")
print("=" * 65)
print(f"Training accuracy  : {train_accuracy * 100:.2f}%")
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
print(f"Test accuracy      : {test_accuracy * 100:.2f}%")
print(
    "Training-validation gap: "
    f"{(train_accuracy - val_accuracy) * 100:.2f} percentage points"
)
if train_accuracy - val_accuracy > 0.10:
    print(
        "Warning: The training-validation gap is large. "
        "Investigate overfitting."
    )
elif val_accuracy - train_accuracy > 0.10:
    print(
        "Note: Validation accuracy exceeds training accuracy "
        "substantially. Dropout and regularization can make "
        "training accuracy lower; inspect the learning curves."
    )
else:
    print(
        "The training-validation accuracy gap is not large. "
        "Also inspect loss curves and per-class metrics."
    )
metrics_df = pd.DataFrame([
    {
        "split": "Training",
        "accuracy": train_results["accuracy"],
        "precision_macro": train_results["precision_macro"],
        "recall_macro": train_results["recall_macro"],
        "f1_macro": train_results["f1_macro"],
    },
    {
        "split": "Validation",
        "accuracy": val_results["accuracy"],
        "precision_macro": val_results["precision_macro"],
        "recall_macro": val_results["recall_macro"],
        "f1_macro": val_results["f1_macro"],
    },
    {
        "split": "Test",
        "accuracy": test_results["accuracy"],
        "precision_macro": test_results["precision_macro"],
        "recall_macro": test_results["recall_macro"],
        "f1_macro": test_results["f1_macro"],
    },
])
metrics_df.to_csv(
    "vit_final_results.csv",
    index=False,
)
pd.DataFrame({
    "class_index": np.arange(len(class_names)),
    "polymer": class_names,
}).to_csv(
    "vit_class_mapping.csv",
    index=False,
)
np.savez(
    "vit_scaler_parameters.npz",
    mean=scaler.mean_,
    scale=scaler.scale_,
)
print("\nMetrics saved to vit_final_results.csv")
plt.figure(figsize=(9, 5))
plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
)
plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
)
plt.title("FTIR Vision Transformer - Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "vit_accuracy.png",
    dpi=300,
)
plt.show()
plt.figure(figsize=(9, 5))
plt.plot(
    history.history["loss"],
    label="Training Loss",
)
plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
)
plt.title("FTIR Vision Transformer - Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "vit_loss.png",
    dpi=300,
)
plt.show()
display = ConfusionMatrixDisplay(
    confusion_matrix=test_results["confusion_matrix"],
    display_labels=class_names,
)
fig, ax = plt.subplots(figsize=(8, 7))
display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)
plt.title("FTIR Vision Transformer - Test Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    "vit_confusion_matrix.png",
    dpi=300,
)
plt.show()
print("\n" + "=" * 65)
print("FTIR VISION TRANSFORMER RUN COMPLETED")
print("=" * 65)
print("Best model       :", MODEL_PATH)
print("Training log     : vit_training_log.csv")
print("Final metrics    : vit_final_results.csv")
print("Class mapping    : vit_class_mapping.csv")
print("Scaler parameters: vit_scaler_parameters.npz")
print("Accuracy plot    : vit_accuracy.png")
print("Loss plot        : vit_loss.png")
print("Confusion matrix : vit_confusion_matrix.png")
print("\nFinal test accuracy:")
print(f"{test_accuracy * 100:.2f}%")