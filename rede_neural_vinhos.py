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

df = pd.read_csv('wine_quality_merged.csv')

# Separação da classe alvo e atributos
X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values 

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
model.add(Dense(10, input_dim=X.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilação do modelo
model.compile(optimizer=SGD(0.01), loss="binary_crossentropy", metrics=["accuracy"])

# Treinamento do modelo
H = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)

# avaliar a Rede Neural
print("[INFO] avaliando a rede neural...")

predictions = model.predict(X_test, batch_size=1)
predictions = (predictions > 0.5).astype(int).flatten()

print(classification_report(y_test, predictions))

print(f'Acurácia: {accuracy:.2f}')

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