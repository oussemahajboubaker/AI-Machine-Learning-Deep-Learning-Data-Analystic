# 👁️ Réseaux de Neurones Convolutifs (CNN) - Vision par Ordinateur

## 🎯 **CONTEXTE ET RÉVOLUTION DE LA VISION PAR ORDINATEUR**

**Qu'est-ce que la Vision par Ordinateur et pourquoi c'est l'avenir ?**

Les CNN (Convolutional Neural Networks) sont **la technologie qui permet aux ordinateurs de "voir" et comprendre les images**. Dans un monde de plus en plus visuel, la capacité à analyser automatiquement les images est **un avantage concurrentiel majeur**.

**Contexte du problème** :
- **Explosion du contenu visuel** : Photos, vidéos, images médicales, satellites...
- **Besoin d'automatisation** : Impossible d'analyser manuellement des millions d'images
- **Découverte d'insights** : Patterns invisibles à l'œil humain
- **Applications critiques** : Diagnostic médical, sécurité, voitures autonomes

**Pourquoi les CNN sont-ils révolutionnaires ?**
- **Inspirés de la vision humaine** : Reproduisent le fonctionnement du cortex visuel
- **Efficacité** : Optimisés pour traiter des données spatiales (images)
- **Performance** : Surpassent les humains dans de nombreuses tâches visuelles
- **Généralisation** : Reconnaissent des objets qu'ils n'ont jamais vus
- **Évolutivité** : Plus d'images = meilleures performances

**Applications qui changent le monde** :
- **Médecine** : Diagnostic par imagerie, détection de cancers
- **Voitures autonomes** : Reconnaissance de panneaux, piétons, obstacles
- **Sécurité** : Reconnaissance faciale, surveillance intelligente
- **E-commerce** : Recherche d'images, recommandations visuelles
- **Agriculture** : Détection de maladies de plantes, optimisation des récoltes
- **Espace** : Analyse d'images satellites, exploration planétaire

## 📚 **CONTENU DU MODULE**

### Notebooks d'Apprentissage Progressifs
- **01-Keras-CNN-MNIST.ipynb** - **Les bases** : Classification de chiffres manuscrits
- **02-Keras-CNN-CIFAR-10.ipynb** - **Objets réels** : Classification d'objets naturels
- **03-Deep-Learning-Images-Réalistes-Malaria.ipynb** - **Médecine** : Diagnostic de malaria
- **04-DL-CV-Exercice.ipynb** - **Pratique** : Exercices de vision par ordinateur

### Dépôt des Exercices
Le dossier `ESPACE DEPOT` contient les exercices soumis par les étudiants.

## 🎯 **OBJECTIFS D'APPRENTISSAGE**

**À la fin de ce module, vous saurez :**
- **Pourquoi** les CNN sont parfaits pour l'analyse d'images
- **Comment** construire des réseaux convolutifs efficaces
- **Comment** préprocesser des images pour l'entraînement
- **Comment** utiliser l'augmentation de données pour améliorer les performances
- **Comment** appliquer le transfer learning sur vos propres images

## 🚀 Prérequis

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
```

## 📖 Concepts Clés

### Architecture CNN de Base

```python
# Modèle CNN simple
model = keras.Sequential([
    # Couche de convolution + pooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Aplatissement et classification
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes pour MNIST
])
```

### Types de Couches CNN

#### Couches de Convolution
```python
# Convolution 2D
layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')

# Convolution avec padding
layers.Conv2D(64, (3, 3), padding='same', activation='relu')

# Convolution avec stride
layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu')
```

#### Couches de Pooling
```python
# Max Pooling
layers.MaxPooling2D(pool_size=(2, 2))

# Average Pooling
layers.AveragePooling2D(pool_size=(2, 2))

# Global Average Pooling
layers.GlobalAveragePooling2D()
```

#### Couches de Normalisation
```python
# Batch Normalization
layers.BatchNormalization()

# Dropout
layers.Dropout(0.5)
```

## 🔧 Exemples Pratiques

### Classification MNIST
```python
# Chargement des données MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocessing
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Conversion des labels en catégories
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Modèle CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compilation
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entraînement
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)
```

### Classification CIFAR-10
```python
# Chargement CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Preprocessing
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Modèle plus complexe pour CIFAR-10
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

### Augmentation de Données
```python
# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Entraînement avec augmentation
model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=50,
    validation_data=(x_test, y_test)
)
```

### Transfer Learning
```python
# Utilisation d'un modèle pré-entraîné
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Geler les couches du modèle de base
base_model.trainable = False

# Ajouter des couches de classification
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

## 📊 Visualisation et Analyse

### Visualisation des Prédictions
```python
# Prédictions sur quelques images
predictions = model.predict(x_test[:10])
predicted_classes = np.argmax(predictions, axis=1)

# Affichage des résultats
plt.figure(figsize=(15, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Prédit: {predicted_classes[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
```

### Matrice de Confusion
```python
# Prédictions complètes
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion')
plt.xlabel('Prédictions')
plt.ylabel('Valeurs Réelles')
plt.show()
```

### Visualisation des Couches
```python
# Extraction des features des couches intermédiaires
layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

# Visualisation des activations
activations = activation_model.predict(x_test[:1])
for i, activation in enumerate(activations):
    plt.figure(figsize=(15, 3))
    for j in range(min(8, activation.shape[-1])):
        plt.subplot(1, 8, j+1)
        plt.imshow(activation[0, :, :, j], cmap='viridis')
        plt.title(f'Filtre {j+1}')
        plt.axis('off')
    plt.suptitle(f'Couche {i+1}')
    plt.tight_layout()
    plt.show()
```

## 💡 Conseils d'Apprentissage

1. **Commencez simple** - MNIST avant CIFAR-10
2. **Normalisez les images** - Divisez par 255.0
3. **Utilisez l'augmentation** - Améliore la généralisation
4. **Expérimentez l'architecture** - Plus de couches ≠ toujours mieux
5. **Monitorer l'overfitting** - Utilisez la validation
6. **Considérez le transfer learning** - Pour des datasets petits

## 🔗 Ressources Supplémentaires

- [Deep Learning for Computer Vision](https://www.pyimagesearch.com/)
- [CS231n Stanford](http://cs231n.stanford.edu/)
- [Keras Applications](https://keras.io/api/applications/)
- [ImageNet](https://www.image-net.org/)
- [Papers with Code - Computer Vision](https://paperswithcode.com/area/computer-vision)
