# 🤖 Machine Learning 1 - Introduction à l'IA Prédictive

## 🎯 **CONTEXTE ET RÉVOLUTION DU MACHINE LEARNING**

**Qu'est-ce que le Machine Learning et pourquoi c'est révolutionnaire ?**

Le Machine Learning (ML) est **l'art de faire apprendre des prédictions aux ordinateurs** à partir de données historiques. C'est comme enseigner à un ordinateur à reconnaître des patterns que même les humains pourraient manquer.

**Contexte historique** : 
- **1950s** : Premiers algorithmes d'apprentissage automatique
- **1990s** : Explosion avec l'Internet et les données massives
- **2010s** : Révolution avec le deep learning et les GPU
- **Aujourd'hui** : L'IA est partout - Netflix, Google, voitures autonomes, médecine...

**Pourquoi le ML est-il révolutionnaire ?**
- **Automatisation** : Faire des prédictions sans programmer explicitement chaque cas
- **Découverte de patterns** : Trouver des corrélations invisibles à l'œil nu
- **Prédiction de l'avenir** : Anticiper les tendances et comportements
- **Optimisation** : Améliorer continuellement les performances
- **Échelle** : Traiter des millions de données simultanément

**Applications qui changent le monde** :
- **Recommandations** : Netflix, Amazon, Spotify
- **Diagnostic médical** : Détection précoce de cancers
- **Voitures autonomes** : Reconnaissance d'objets et prédiction de trajectoires
- **Finance** : Détection de fraude, trading algorithmique
- **Climat** : Prévisions météo, modélisation du changement climatique

## 📚 **CONTENU DU MODULE**

### Notebooks d'Apprentissage
- **Machine Learning 1.ipynb** - **Introduction complète** : Concepts, types, et applications du ML

### Datasets d'Exemple Temporels
- `ave_hi_nyc_jan_1895-2018.csv` - **Données climatiques** : Températures NYC (123 ans de données !)
- `ave_yearly_temp_nyc_1895-2017.csv` - **Série temporelle** : Évolution des températures annuelles

## 🎯 **OBJECTIFS D'APPRENTISSAGE**

**À la fin de ce module, vous saurez :**
- **Pourquoi** le ML est la technologie la plus importante de notre époque
- **Comment** distinguer les différents types d'apprentissage
- **Quand** utiliser le ML vs les méthodes traditionnelles
- **Comment** structurer un projet de ML de A à Z
- **Comment** évaluer si un modèle est bon ou mauvais

## 🚀 Prérequis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

## 📖 Concepts Clés

### Types d'Apprentissage

#### Apprentissage Supervisé
- **Régression** - Prédire une valeur continue
- **Classification** - Prédire une catégorie/classe

#### Apprentissage Non-Supervisé
- **Clustering** - Grouper des données similaires
- **Réduction de dimensionnalité** - Simplifier les données

### Pipeline de Machine Learning

```python
# 1. Chargement et exploration des données
df = pd.read_csv('data.csv')
df.head()
df.describe()
df.info()

# 2. Préparation des données
X = df[['feature1', 'feature2']]
y = df['target']

# 3. Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prédictions
y_pred = model.predict(X_test)

# 6. Évaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### Algorithmes de Base

#### Régression Linéaire
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### Régression Polynomiale
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])
```

### Évaluation des Modèles

```python
# Métriques de régression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualisation des résultats
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions')
plt.title('Régression Linéaire')
```

## 📊 Analyse des Données Temporelles

### Série Temporelle
```python
# Chargement des données temporelles
df = pd.read_csv('temperature_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Visualisation
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Temperature'])
plt.title('Évolution de la Température')
plt.xlabel('Date')
plt.ylabel('Température (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Détection de Tendances
```python
# Moyenne mobile
df['MA_30'] = df['Temperature'].rolling(window=30).mean()

# Tendance linéaire
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(df)), df['Temperature'])
```

## 💡 Conseils d'Apprentissage

1. **Comprenez vos données** - Exploration et visualisation sont cruciales
2. **Divisez correctement** - Train/validation/test sets appropriés
3. **Évaluez objectivement** - Utilisez les bonnes métriques
4. **Commencez simple** - Modèles linéaires avant les complexes
5. **Validez vos résultats** - Cross-validation et tests robustes

## 🔗 Ressources Supplémentaires

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Coursera ML Course](https://www.coursera.org/learn/machine-learning)
- [Hands-On Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
