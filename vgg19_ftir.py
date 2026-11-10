import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "FTIR-PLASTIC/FTIR-PLASTIC-c8/FTIR_PLASTIC_c8.csv"
)

# Get FTIR intensity columns
y_columns = [
    c for c in df.columns
    if c.startswith("Data(Y)")
]
X = df[y_columns].values
y = df["Polymer"].values
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print("Classes:")
print(encoder.classes_)
print("\nClass distribution:")
unique, counts = np.unique(y, return_counts=True)
for class_id, count in zip(unique, counts):
    print(
        encoder.classes_[class_id],
        ":",
        count
    )
X = X.astype("float32")
X = (X - X.min()) / (X.max() - X.min())
size = 44
total_values = size * size
X_padded = np.zeros(
    (X.shape[0], total_values),
    dtype="float32"
)
X_padded[:, :X.shape[1]] = X
X = X_padded.reshape(
    X.shape[0],
    size,
    size,
    1
)
X = np.repeat(
    X,
    3,
    axis=-1
)
print("\nInput shape:")
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))
def vgg_block(x, filters, number_of_convs):
    for i in range(number_of_convs):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding="same",
            activation="relu"
        )(x)
        x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(x)
    return x
inputs = layers.Input(
    shape=(44, 44, 3)
)
x = vgg_block(inputs,64,2)
x = vgg_block(x,128,2)
x = vgg_block(x,256,4)
x = vgg_block(x,512,4)
x = vgg_block(x,512,4)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256,activation="relu")(x)
x = layers.Dropout(
    0.5
)(x)
outputs = layers.Dense(
    6,
    activation="softmax"
)(x)
model = Model(
    inputs=inputs,
    outputs=outputs
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    shuffle=True
)
y_probability = model.predict(
    X_test
)
y_pred = np.argmax(
    y_probability,
    axis=1
)
accuracy = accuracy_score(
    y_test,
    y_pred
)

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


# ============================================================
# 16. PRINT FINAL RESULTS
# ============================================================

print("\n")
print("================================")
print("FINAL RESULTS")
print("================================")

print(
    "Accuracy :",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall   :",
    round(recall, 4)
)

print(
    "F1 Score :",
    round(f1, 4)
)
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n")
print("================================")
print("CONFUSION MATRIX")
print("================================")

print(cm)


# ============================================================
# 18. CLASS ORDER
# ============================================================

print("\n")
print("================================")
print("CLASS ORDER")
print("================================")

for i, class_name in enumerate(
    encoder.classes_
):

    print(
        i,
        "=",
        class_name
    )