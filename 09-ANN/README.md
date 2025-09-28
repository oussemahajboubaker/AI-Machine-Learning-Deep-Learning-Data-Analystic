# 🧠 Réseaux de Neurones Artificiels (ANN) - L'Intelligence Artificielle

## 🎯 **CONTEXTE ET RÉVOLUTION DU DEEP LEARNING**

**Qu'est-ce que le Deep Learning et pourquoi c'est révolutionnaire ?**

Les Réseaux de Neurones Artificiels (ANN) sont **l'épine dorsale de l'Intelligence Artificielle moderne**. Inspirés du fonctionnement du cerveau humain, ils peuvent apprendre des patterns complexes que les algorithmes traditionnels ne peuvent pas détecter.

**Contexte historique** :
- **1940s** : Premiers modèles de neurones artificiels
- **1980s** : Perceptron et premiers réseaux multicouches
- **2000s** : Révolution avec le deep learning et les GPU
- **2010s** : Explosion avec TensorFlow, PyTorch, et les données massives
- **Aujourd'hui** : L'IA est partout - reconnaissance d'images, traduction, voitures autonomes...

**Pourquoi le Deep Learning est-il révolutionnaire ?**
- **Apprentissage automatique** : Découvre des patterns sans programmation explicite
- **Capacité d'abstraction** : Comprend des concepts complexes
- **Généralisation** : Applique ce qu'il apprend à de nouvelles situations
- **Performance** : Surpasse les humains dans de nombreux domaines
- **Évolutivité** : Plus de données = meilleures performances

**Applications qui changent le monde** :
- **Reconnaissance d'images** : Photos, vidéos, imagerie médicale
- **Traitement du langage** : Traduction, chatbots, assistants vocaux
- **Voitures autonomes** : Reconnaissance d'objets, prédiction de trajectoires
- **Médecine** : Diagnostic, découverte de médicaments, imagerie
- **Finance** : Trading algorithmique, détection de fraude
- **Jeux** : AlphaGo, jeux vidéo intelligents

## 📚 **CONTENU DU MODULE**

### Notebooks d'Apprentissage Progressifs
- **00-Bases-Syntaxe-Keras.ipynb** - **Les fondations** : Syntaxe Keras et TensorFlow 2.x
- **01-Régression-Keras.ipynb** - **Prédiction continue** : Réseaux pour la régression
- **02-Classification-Keras.ipynb** - **Classification intelligente** : Réseaux pour la classification
- **03-Exercice-Projet-Keras.ipynb** - **Projet pratique** : Application complète
- **05-Tensorboard.ipynb** - **Monitoring** : Visualisation et suivi des performances

### Datasets d'Exemple Réels
- `cancer_classification.csv` - **Données médicales** : Classification de cancer
- `fake_reg.csv` - **Données simulées** : Régression simple pour apprendre
- `kc_house_data.csv` - **Données immobilières** : Prédiction de prix de maisons
- `lending_club_info.csv` - **Données financières** : Informations sur les prêts
- `lending_club_loan_two.csv` - **Données de prêts** : Classification de risque

## 🎯 **OBJECTIFS D'APPRENTISSAGE**

**À la fin de ce module, vous saurez :**
- **Pourquoi** le deep learning est la technologie la plus importante de notre époque
- **Comment** construire des réseaux de neurones efficaces
- **Comment** choisir l'architecture appropriée pour votre problème
- **Comment** éviter l'overfitting et optimiser les performances
- **Comment** utiliser TensorBoard pour monitorer vos modèles

## 🚀 Prérequis

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
```

## 📖 Concepts Clés

### Architecture d'un Réseau de Neurones

```python
# Création d'un modèle séquentiel
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  # Classification binaire
])

# Compilation du modèle
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Types de Couches

#### Couches Denses (Fully Connected)
```python
layers.Dense(units=64, activation='relu')
```

#### Couches de Dropout (Régularisation)
```python
layers.Dropout(rate=0.2)
```

#### Couches de Normalisation
```python
layers.BatchNormalization()
```

### Fonctions d'Activation

```python
# Fonctions d'activation courantes
'relu'      # Rectified Linear Unit
'sigmoid'   # Pour la classification binaire
'softmax'   # Pour la classification multi-classe
'tanh'      # Tangente hyperbolique
'linear'    # Pas d'activation (régression)
```

### Optimiseurs

```python
# Optimiseurs disponibles
'adam'      # Adaptive Moment Estimation (recommandé)
'sgd'       # Stochastic Gradient Descent
'rmsprop'   # Root Mean Square Propagation
'adagrad'   # Adaptive Gradient
```

## 🔧 Exemples Pratiques

### Régression avec Keras
```python
# Préparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle de régression
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)  # Pas d'activation pour la régression
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Entraînement
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

### Classification avec Keras
```python
# Modèle de classification
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Classification binaire
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Entraînement
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    callbacks=[keras.callbacks.EarlyStopping(patience=5)]
)
```

### Visualisation des Résultats
```python
# Historique d'entraînement
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```

## 🎛️ Techniques Avancées

### Callbacks
```python
# Callbacks utiles
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]
```

### Régularisation
```python
# L1 et L2 regularization
layers.Dense(64, activation='relu', 
             kernel_regularizer=keras.regularizers.l2(0.01))

# Dropout
layers.Dropout(0.3)

# Batch Normalization
layers.BatchNormalization()
```

### TensorBoard
```python
# Configuration TensorBoard
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# Utilisation dans l'entraînement
model.fit(X_train, y_train, callbacks=[tensorboard_callback])
```

## 💡 Conseils d'Apprentissage

1. **Commencez simple** - Modèles simples avant les complexes
2. **Normalisez vos données** - Crucial pour la convergence
3. **Utilisez la validation** - Évitez l'overfitting
4. **Expérimentez** - Testez différentes architectures
5. **Monitorer** - Utilisez TensorBoard pour visualiser
6. **Régularisez** - Dropout et Batch Normalization

## 🔗 Ressources Supplémentaires

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
