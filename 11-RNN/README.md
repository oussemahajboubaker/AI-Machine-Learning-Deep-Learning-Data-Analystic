# ⏰ Réseaux de Neurones Récurrents (RNN) - Prédiction du Futur

## 🎯 **CONTEXTE ET POUVOIR DE LA PRÉDICTION TEMPORELLE**

**Qu'est-ce que les RNN et pourquoi c'est le graal de l'analyse de données ?**

Les RNN (Recurrent Neural Networks) sont **la technologie qui permet de prédire l'avenir** à partir du passé. Dans un monde où **le temps est la dimension la plus importante**, savoir anticiper les tendances est **une compétence superpuissance**.

**Contexte du problème** :
- **Données temporelles partout** : Prix des actions, ventes, température, trafic, santé...
- **Besoin de prédiction** : Anticiper pour optimiser, prévenir, planifier
- **Complexité temporelle** : Les patterns changent dans le temps
- **Dépendances séquentielles** : Le futur dépend du passé

**Pourquoi les RNN sont-ils révolutionnaires ?**
- **Mémoire** : Se souviennent des informations passées
- **Séquentiel** : Traitent les données dans l'ordre chronologique
- **Prédiction** : Anticipent les valeurs futures
- **Adaptabilité** : Apprennent des patterns temporels complexes
- **Généralisation** : Appliquent ce qu'ils apprennent à de nouvelles séquences

**Applications qui changent le monde** :
- **Finance** : Prédiction des prix, trading algorithmique, gestion des risques
- **Météo** : Prévisions météorologiques, modélisation climatique
- **Médecine** : Prédiction d'épidémies, monitoring de patients
- **Transport** : Optimisation du trafic, maintenance prédictive
- **Énergie** : Prédiction de consommation, optimisation des réseaux
- **E-commerce** : Prédiction de ventes, gestion des stocks

## 📚 **CONTENU DU MODULE**

### Notebooks d'Apprentissage Progressifs
- **00-RNN-Exemple-Sinusoïdale.ipynb** - **Les bases** : Introduction avec une onde sinusoïdale simple
- **01-RNN-Exemple-Time-Series.ipynb** - **Séries réelles** : Analyse de séries temporelles complexes
- **02-RNN-Exercice.ipynb** - **Pratique** : Exercices pour maîtriser les RNN
- **04-BONUS-RNN-Multivarié.ipynb** - **Avancé** : RNN multivariés pour des données complexes

### Datasets d'Exemple Temporels
- `energydata_complete.csv` - **Données énergétiques** : Consommation d'énergie dans le temps
- `Frozen_Dessert_Production.csv` - **Production industrielle** : Production de desserts glacés
- `MRTSSM448USN.csv` - **Données économiques** : Ventes au détail
- `RSCCASN.csv` - **Série temporelle** : Données de vente au détail
- `RSCCASN2021.csv` - **Données 2021** : Évolution récente
- `RSCCASN2023.csv` - **Données 2023** : Tendances actuelles

## 🎯 **OBJECTIFS D'APPRENTISSAGE**

**À la fin de ce module, vous saurez :**
- **Pourquoi** les RNN sont parfaits pour les données temporelles
- **Comment** prédire l'avenir à partir de données historiques
- **Comment** choisir entre Simple RNN, LSTM, et GRU
- **Comment** préprocesser des données temporelles
- **Comment** évaluer la qualité de vos prédictions

## 🚀 Prérequis

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
```

## 📖 Concepts Clés

### Architecture RNN de Base

```python
# Modèle RNN simple
model = keras.Sequential([
    layers.SimpleRNN(50, return_sequences=True, input_shape=(timesteps, features)),
    layers.SimpleRNN(50, return_sequences=False),
    layers.Dense(25),
    layers.Dense(1)
])
```

### Types de RNN

#### Simple RNN
```python
layers.SimpleRNN(units=50, activation='tanh')
```

#### LSTM (Long Short-Term Memory)
```python
layers.LSTM(units=50, return_sequences=True)
```

#### GRU (Gated Recurrent Unit)
```python
layers.GRU(units=50, return_sequences=True)
```

## 🔧 Exemples Pratiques

### Prédiction de Série Temporelle Simple
```python
# Préparation des données
def create_sequences(data, timesteps):
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Création des séquences
timesteps = 60
X, y = create_sequences(scaled_data, timesteps)

# Division train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Modèle LSTM
model = keras.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(timesteps, 1)),
    layers.Dropout(0.2),
    layers.LSTM(50, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(50),
    layers.Dropout(0.2),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
```

### Prédiction Multivariée
```python
# Modèle pour données multivariées
model = keras.Sequential([
    layers.LSTM(50, return_sequences=True, input_shape=(timesteps, n_features)),
    layers.Dropout(0.2),
    layers.LSTM(50, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(25),
    layers.Dense(1)
])
```

## 📊 Visualisation des Résultats

```python
# Prédictions vs Valeurs réelles
plt.figure(figsize=(15, 6))
plt.plot(y_test, label='Valeurs Réelles', alpha=0.7)
plt.plot(predictions, label='Prédictions', alpha=0.7)
plt.title('Prédiction de Série Temporelle')
plt.xlabel('Temps')
plt.ylabel('Valeur')
plt.legend()
plt.show()
```

## 💡 Conseils d'Apprentissage

1. **Comprenez la séquence** - Les RNN sont conçus pour les données temporelles
2. **Choisissez le bon type** - LSTM pour la mémoire longue, GRU pour la simplicité
3. **Normalisez les données** - Crucial pour la convergence
4. **Expérimentez les timesteps** - Plus long ≠ toujours mieux
5. **Utilisez la régularisation** - Dropout pour éviter l'overfitting

## 🔗 Ressources Supplémentaires

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Deep Learning for Time Series](https://www.manning.com/books/deep-learning-for-time-series-forecasting)
- [TensorFlow RNN Guide](https://www.tensorflow.org/guide/keras/rnn)
