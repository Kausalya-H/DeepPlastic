import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU,
    MaxPooling2D, GlobalAveragePooling2D,
    Dense, Concatenate, AveragePooling2D
)
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
df = pd.read_csv(
    "FTIR-PLASTIC/FTIR-PLASTIC-c8/FTIR_PLASTIC_c8.csv"
)
print("Dataset shape:", df.shape)

y_columns = [
    c for c in df.columns
    if c.startswith("Data(Y)")
]

X = df[y_columns].values
y = df["Polymer"].values
print("Number of spectral points:", len(y_columns))
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print("\nClasses:")
print(encoder.classes_)
X = X.astype("float32")
X = (X - X.min()) / (X.max() - X.min())
size = 44
total_values = size * size
X_padded = np.zeros((X.shape[0], total_values),dtype="float32")
X_padded[:, :X.shape[1]] = X

X = X_padded.reshape(X.shape[0],size,size,1)
X = np.repeat(X,3,axis=-1)
print("Input shape:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
def dense_block(x, filters, blocks):
    for i in range(blocks):
        previous = x
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters * 4,kernel_size=(1, 1),padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters,kernel_size=(3, 3),padding="same")(x)
        x = Concatenate()([previous,x])
    return x
def transition_block(x, filters):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters,kernel_size=(1, 1),padding="same")(x)
    x = AveragePooling2D(pool_size=(2, 2),strides=(2, 2))(x)
    return x
input_layer = Input(shape=(44, 44, 3))
x = Conv2D(64,kernel_size=(7, 7),strides=(2, 2),padding="same")(input_layer)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D(pool_size=(3, 3),strides=(2, 2),padding="same")(x)
x = dense_block(x,filters=32,blocks=6)
x = transition_block(x,filters=128)
x = dense_block(x,filters=32,blocks=12)
x = transition_block(x,filters=256)
x = dense_block(x,filters=32,blocks=24)
x = transition_block(x,filters=512)
x = dense_block(x,filters=32,blocks=16)
x = BatchNormalization()(x)
x = ReLU()(x)
x = GlobalAveragePooling2D()(x)
output_layer = Dense(6,activation="softmax"
)(x)
model = Model(inputs=input_layer,outputs=output_layer)
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
print("\n================================")
print("DENSENET-121 MODEL")
print("================================")
model.summary()
model.fit(X_train,y_train,epochs=10,batch_size=32,validation_split=0.2)
predictions = model.predict(X_test)
y_pred = np.argmax(
    predictions,
    axis=1
)
accuracy = accuracy_score(
    y_test,
    y_pred
)
precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)
f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)
print("\n================================")
print("FINAL RESULTS")
print("================================")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
cm = confusion_matrix(
    y_test,
    y_pred
)
print("\nConfusion Matrix:")
print(cm)
print("\nClass order:")
print(encoder.classes_)