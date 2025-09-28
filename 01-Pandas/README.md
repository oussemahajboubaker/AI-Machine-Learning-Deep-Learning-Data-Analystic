# 📊 Pandas - L'Excel des Data Scientists

## 🎯 **CONTEXTE ET RÉVOLUTION DE PANDAS**

**Qu'est-ce que Pandas et pourquoi c'est révolutionnaire ?**

Pandas est **LA bibliothèque Python** qui transforme Python en un outil d'analyse de données plus puissant qu'Excel. Créé en 2008 par Wes McKinney, Pandas a démocratisé l'analyse de données en Python.

**Contexte du problème** : Avant Pandas, analyser des données en Python était un cauchemar :
- Manipulation manuelle de listes et dictionnaires
- Calculs lents et peu intuitifs
- Pas de structure claire pour les données tabulaires
- Difficile de gérer les données manquantes

**La révolution Pandas** :
- **80% du travail d'un data scientist** consiste à nettoyer et explorer des données
- Pandas rend cela **intuitif et efficace**
- Interface similaire à Excel mais **1000x plus puissante**
- **Base indispensable** pour tout projet de machine learning

**Pourquoi Pandas est crucial ?**
- **Préparation des données** : 80% du temps en data science
- **Exploration** : Comprendre vos données avant de les modéliser
- **Nettoyage** : Gérer les données manquantes, outliers, formats
- **Transformation** : Créer de nouvelles variables, agréger, pivoter

## 📚 **CONTENU DU MODULE**

### Notebooks d'Apprentissage Progressifs
- **00-Introduction-à-Pandas.ipynb** - **Les concepts** : Series vs DataFrames, pourquoi Pandas ?
- **01-Series-Pandas.ipynb** - **Données 1D** : Manipulation des Series (comme des colonnes Excel)
- **02-DataFrames-Pandas.ipynb** - **Données 2D** : Manipulation des DataFrames (comme des feuilles Excel)
- **03-Données-Manquantes.ipynb** - **Le nettoyage** : Gérer les valeurs manquantes (très fréquent !)
- **04-GroupBy.ipynb** - **L'agrégation** : Grouper et résumer les données
- **05-Opérations.ipynb** - **Les transformations** : Opérations avancées sur les données
- **06-Data-Input-et-Output.ipynb** - **Import/Export** : Charger et sauvegarder des données
- **07-Exercices-Pandas.ipynb** - **La pratique** : Exercices pour maîtriser Pandas

### Datasets d'Exemple Réels
- `african_econ_crises.csv` - **Données économiques** : Crises économiques africaines
- `bank.csv` - **Données bancaires** : Informations clients bancaires
- `example.csv` - **Dataset d'exemple** : Pour les premiers pas
- `Universities.csv` - **Données éducatives** : Classements d'universités

## 🎯 **OBJECTIFS D'APPRENTISSAGE**

**À la fin de ce module, vous saurez :**
- **Pourquoi** Pandas est indispensable en data science
- **Comment** manipuler des données comme un pro
- **Quand** utiliser Series vs DataFrames
- **Comment** nettoyer et préparer des données pour le ML
- **Comment** explorer efficacement vos datasets
- **Comment** transformer des données pour l'analyse

## 🚀 Prérequis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## 📖 Concepts Clés

### Series
```python
# Création d'une Series
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])

# Accès aux données
s['a']           # Valeur à l'index 'a'
s[0:3]           # Slice des 3 premiers éléments
s.describe()     # Statistiques descriptives
```

### DataFrames
```python
# Création d'un DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40],
    'C': ['a', 'b', 'c', 'd']
})

# Accès aux colonnes et lignes
df['A']                    # Colonne A
df[['A', 'B']]             # Colonnes A et B
df.loc[0]                  # Ligne d'index 0
df.iloc[0:2]               # Lignes 0 et 1
```

### Gestion des Données Manquantes
```python
# Détection des valeurs manquantes
df.isnull()
df.isnull().sum()

# Suppression des valeurs manquantes
df.dropna()                # Supprime les lignes avec NaN
df.dropna(axis=1)          # Supprime les colonnes avec NaN

# Remplissage des valeurs manquantes
df.fillna(0)               # Remplit avec 0
df.fillna(df.mean())       # Remplit avec la moyenne
```

### Opérations de Groupement
```python
# GroupBy
df.groupby('colonne').mean()           # Moyenne par groupe
df.groupby('colonne').agg(['mean', 'std'])  # Plusieurs fonctions d'agrégation

# Pivot tables
df.pivot_table(values='A', index='B', columns='C', aggfunc='mean')
```

### Lecture et Écriture de Fichiers
```python
# Lecture
df = pd.read_csv('fichier.csv')
df = pd.read_excel('fichier.xlsx')
df = pd.read_json('fichier.json')

# Écriture
df.to_csv('sortie.csv', index=False)
df.to_excel('sortie.xlsx', index=False)
```

## 💡 Conseils d'Apprentissage

1. **Pratiquez avec des données réelles** - Utilisez les datasets fournis
2. **Maîtrisez l'indexation** - Crucial pour la manipulation de données
3. **Comprenez les opérations vectorisées** - Plus rapides que les boucles
4. **Apprenez à gérer les données manquantes** - Très fréquent en pratique
5. **Utilisez les méthodes de chaînage** - Code plus lisible et efficace

## 🔗 Ressources Supplémentaires