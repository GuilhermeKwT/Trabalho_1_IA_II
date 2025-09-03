from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.metrics import precision_score, f1_score

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.precisions = []
        self.f1s = []

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        val_predict = (self.model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
        precision = precision_score(y_val, val_predict)
        f1 = f1_score(y_val, val_predict)
        self.precisions.append(precision)
        self.f1s.append(f1)

df = pd.read_csv('wine_quality_merged (2).csv')

# Separação da classe alvo e atributos
X = df.drop('type', axis=1)
y = df['type']

# Transformação das classes para valores numéricos (0 e 1)
le = LabelEncoder()
y = le.fit_transform(y)

# Normalização dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Definição do modelo
model = Sequential()
model.add(Dense(11, input_dim=X.shape[1], activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilação do modelo
model.compile(optimizer=SGD(0.01), loss="binary_crossentropy", metrics=["accuracy"])

# Treinamento do modelo
metrics_callback = MetricsCallback(validation_data=(X_test, y_test))
H = model.fit(X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[metrics_callback])

loss, accuracy = model.evaluate(X_test, y_test)

# avaliar a Rede Neural
print("[INFO] avaliando a rede neural...")

predictions = model.predict(X_test, batch_size=1)
predictions = (predictions > 0.5).astype(int).flatten()

print(classification_report(y_test, predictions, digits=6))

print(f'Acurácia: {accuracy:.6f}')

results_df = pd.DataFrame({
    'epoch': np.arange(1, 101),
    'val_precision': metrics_callback.precisions,
    'val_f1': metrics_callback.f1s
})
print(results_df)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()