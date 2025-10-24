from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import pandas as pd

from sklearn.naive_bayes import LabelBinarizer
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
        val_pred = self.model.predict(X_val, verbose=0)
        val_pred_classes = val_pred.argmax(axis=1)
        y_val_classes = y_val.argmax(axis=1)
        precision = precision_score(y_val_classes, val_pred_classes, average='macro', zero_division=0)
        f1 = f1_score(y_val_classes, val_pred_classes, average='macro', zero_division=0)
        self.precisions.append(precision)
        self.f1s.append(f1)

df = pd.read_csv('wine_quality_merged_quality.csv')

# Separação da classe alvo e atributos
X = df.drop('quality', axis=1)
y = df['quality']

# Normalização dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# Definição do modelo
model = Sequential()
model.add(Dense(11, input_dim=X.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='softmax'))

# Compilação do modelo
model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy", metrics=["accuracy"])

# Treinamento do modelo
metrics_callback = MetricsCallback(validation_data=(X_test, y_test))
H = model.fit(X_train, y_train,
    epochs=150,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[metrics_callback])

loss, accuracy = model.evaluate(X_test, y_test)

# avaliar a Rede Neural
print("[INFO] avaliando a rede neural...")

predictions = model.predict(X_test)
pred_classes = predictions.argmax(axis=1)
true_classes = y_test.argmax(axis=1)

print(classification_report(true_classes, pred_classes, digits=6))

print(f'Acurácia: {accuracy:.6f}')

results_df = pd.DataFrame({
    'epoch': np.arange(1, 151),
    'val_precision': metrics_callback.precisions,
    'val_f1': metrics_callback.f1s
})
print(results_df)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,150), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,150), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,150), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0,150), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
