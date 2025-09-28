# 🎯 Naive Bayes - Classification Intelligente de Texte

## 🎯 **CONTEXTE ET RÉVOLUTION DE LA CLASSIFICATION DE TEXTE**

**Qu'est-ce que Naive Bayes et pourquoi c'est révolutionnaire pour le texte ?**

Naive Bayes est **l'algorithme de classification probabiliste** le plus efficace pour analyser du texte. Dans notre ère digitale où **des millions de textes** sont générés chaque seconde, savoir les analyser automatiquement est **une compétence superpuissance**.

**Contexte du problème** :
- **Explosion du contenu textuel** : Emails, tweets, avis clients, articles, messages...
- **Besoin d'automatisation** : Impossible de lire manuellement des millions de textes
- **Classification cruciale** : Spam vs important, positif vs négatif, urgent vs normal
- **Découverte d'insights** : Comprendre les sentiments et tendances des clients

**Pourquoi Naive Bayes est-il révolutionnaire ?**
- **Simplicité** : Facile à comprendre et implémenter
- **Efficacité** : Très rapide même sur de gros volumes
- **Performance** : Excellent pour la classification de texte
- **Probabilités** : Fournit des scores de confiance
- **Robustesse** : Fonctionne bien même avec peu de données

**Applications qui changent le monde** :
- **Filtrage de spam** : Gmail, Outlook (milliards d'emails filtrés)
- **Analyse de sentiment** : Comprendre l'opinion des clients
- **Chatbots** : Classifier les intentions des utilisateurs
- **Réseaux sociaux** : Modération automatique de contenu
- **E-commerce** : Recommandations basées sur les avis

## 📚 **CONTENU DU MODULE**

### Notebooks d'Apprentissage
- **Naive bayes.ipynb** - **Introduction complète** : De la théorie à la pratique

### Datasets d'Exemple Réels
- `smsspamcollection/SMSSpamCollection` - **Collection SMS** : Messages spam vs légitimes

### Dépôt des Exercices
Le dossier `ESPACE DEPOT` contient les exercices soumis par les étudiants.

### Ressources Visuelles Pédagogiques
- `bayes_formula.png` - **Formule de Bayes** : La base mathématique
- `countvectorizer.png` - **CountVectorizer** : Comment transformer le texte en nombres
- `dqnb.png` - **Diagramme Naive Bayes** : Visualisation de l'algorithme
- `naivebayes.png` - **Représentation visuelle** : Comprendre le processus
- `tfidf.png` - **TF-IDF** : Pondération des mots importants

## 🎯 **OBJECTIFS D'APPRENTISSAGE**

**À la fin de ce module, vous saurez :**
- **Pourquoi** Naive Bayes est parfait pour l'analyse de texte
- **Comment** transformer du texte en données numériques
- **Comment** classifier automatiquement des milliers de documents
- **Comment** analyser les sentiments des clients
- **Comment** créer un filtre anti-spam intelligent

## 🚀 Prérequis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
```

## 📖 Concepts Clés

### Théorème de Bayes
```
P(A|B) = P(B|A) * P(A) / P(B)
```

Où :
- P(A|B) = Probabilité a posteriori
- P(B|A) = Vraisemblance
- P(A) = Probabilité a priori
- P(B) = Probabilité marginale

### Naive Bayes
L'algorithme Naive Bayes suppose que les features sont conditionnellement indépendants :

```
P(y|x1,x2,...,xn) = P(y) * ∏ P(xi|y)
```

### Types de Naive Bayes

#### Multinomial Naive Bayes
```python
# Pour les données de comptage (texte)
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### Gaussian Naive Bayes
```python
# Pour les données continues
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## 🔤 Prétraitement de Texte

### Tokenization et Nettoyage
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer la ponctuation et les chiffres
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Supprimer les stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)
```

### Vectorisation de Texte

#### Count Vectorizer
```python
from sklearn.feature_extraction.text import CountVectorizer

# Créer le vectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')

# Transformer les données
X = cv.fit_transform(text_data)
feature_names = cv.get_feature_names_out()
```

#### TF-IDF Vectorizer
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Créer le vectorizer TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

# Transformer les données
X = tfidf.fit_transform(text_data)
```

## 📊 Classification de Sentiment

### Exemple Complet
```python
# Chargement des données
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])

# Prétraitement
df['processed_text'] = df['message'].apply(preprocess_text)

# Vectorisation
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['processed_text'])
y = df['label']

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = MultinomialNB()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### Visualisation des Résultats
```python
# Matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion')
plt.xlabel('Prédictions')
plt.ylabel('Valeurs Réelles')
plt.show()

# Importance des mots
feature_names = vectorizer.get_feature_names_out()
feature_importance = model.feature_log_prob_[1] - model.feature_log_prob_[0]
top_words = np.argsort(feature_importance)[-20:]

plt.figure(figsize=(10, 8))
plt.barh(range(len(top_words)), feature_importance[top_words])
plt.yticks(range(len(top_words)), [feature_names[i] for i in top_words])
plt.title('Mots les Plus Importants pour la Classification')
plt.xlabel('Importance')
plt.show()
```

## 💡 Avantages et Inconvénients

### Avantages
- **Simplicité** - Facile à implémenter et comprendre
- **Rapidité** - Entraînement et prédiction très rapides
- **Efficacité** - Fonctionne bien avec peu de données
- **Probabilités** - Fournit des probabilités de classe

### Inconvénients
- **Hypothèse naïve** - L'indépendance des features est rarement vraie
- **Sensibilité** - Peut être sensible aux features corrélées
- **Qualité des données** - Nécessite un bon prétraitement

## 🎯 Applications Pratiques

- **Filtrage de spam** - Classification des emails
- **Analyse de sentiment** - Opinion sur les produits/services
- **Classification de documents** - Catégorisation automatique
- **Détection de fraude** - Identification des transactions suspectes

## 🔗 Ressources Supplémentaires

- [Scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [NLTK Documentation](https://www.nltk.org/)
- [Text Classification with Naive Bayes](https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/)
- [Natural Language Processing with Python](https://www.nltk.org/book/)
