import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, InputLayer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import save_model

# Créer un dossier pour sauvegarder les modèles
os.makedirs("models", exist_ok=True)

# Charger et préparer les données
X, y = load_iris(return_X_y=True)
y_cat = to_categorical(y)  # Pour le CNN

# Normalisation pour SVM et CNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ➤ Linear Support Vector Classifier
linear_svc = LinearSVC(max_iter=10000)
linear_svc.fit(X_scaled, y)
joblib.dump(linear_svc, "models/linear_svc.pkl")

# ➤ Kernel SVC
kernel_svc = SVC(kernel='rbf', probability=True)
kernel_svc.fit(X_scaled, y)
joblib.dump(kernel_svc, "models/kernel_svc.pkl")

# ➤ Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X, y)
joblib.dump(tree, "models/decision_tree.pkl")

# ➤ Random Forest
forest = RandomForestClassifier()
forest.fit(X, y)
joblib.dump(forest, "models/random_forest.pkl")

# ➤ Convolutional Neural Network (sur données tabulaires 1D)
# Reshape pour Conv1D : (n_samples, n_features, 1)
X_cnn = np.expand_dims(X_scaled, axis=-1)

cnn = Sequential([
    InputLayer(input_shape=(X_cnn.shape[1], 1)),
    Conv1D(32, kernel_size=2, activation='relu'),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes pour Iris
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_cnn, y_cat, epochs=20, verbose=0)
save_model(cnn, "models/cnn_model.keras")

print("\n✅ Tous les modèles ont été entraînés et sauvegardés.")
