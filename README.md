# ⚽ Prédiction de Résultats de Football - Ligue 1 🏆

Une application Streamlit pour prédire les résultats des matchs de football de la Ligue 1 en utilisant l'apprentissage automatique.

## 🔗 Application en ligne

Essayez l'application directement dans votre navigateur:
[https://blank-app-0usz19xzsyoo.streamlit.app/](https://blank-app-0usz19xzsyoo.streamlit.app/)

## 🛠️ Options et fonctionnalités

### 🚀 Mode avancé

Le mode avancé débloque des fonctionnalités supplémentaires pour les utilisateurs expérimentés :

- ✅ Activation de la validation croisée
- ✅ Paramètres supplémentaires pour les modèles (min_samples_leaf pour Random Forest)
- ✅ Affichage d'informations plus détaillées sur les modèles
- ✅ Calcul d'importance des caractéristiques par permutation pour SVM

### 📊 Utiliser des caractéristiques avancées

Cette option permet d'inclure des statistiques détaillées des matchs en plus des cotes des bookmakers :

- ⚽ FTHG/FTAG : Nombre de buts marqués à domicile/extérieur
- 🎯 HS/AS : Nombre de tirs à domicile/extérieur
- 🎯 HST/AST : Nombre de tirs cadrés à domicile/extérieur
- 🚩 HC/AC : Nombre de corners à domicile/extérieur
- 🤼 HF/AF : Nombre de fautes commises à domicile/extérieur
- 🟨 HY/AY : Nombre de cartons jaunes à domicile/extérieur
- 🟥 HR/AR : Nombre de cartons rouges à domicile/extérieur

En activant cette option, le modèle utilisera plus d'informations pour faire ses prédictions, ce qui peut améliorer sa précision mais augmente le risque de surapprentissage.

### 🔄 Validation croisée

La validation croisée est une technique qui permet d'évaluer la performance du modèle de manière plus robuste :

- 🧩 Les données sont divisées en plusieurs partitions (folds)
- 🔁 Le modèle est entraîné et évalué plusieurs fois sur différentes combinaisons de ces partitions
- 📈 Les résultats sont moyennés pour obtenir une estimation plus fiable de la performance du modèle

Cette technique est particulièrement utile lorsque la quantité de données est limitée, car elle permet d'utiliser efficacement toutes les données disponibles.

### ⚡ Activer toutes les options

Si vous activez le mode avancé, la validation croisée et les caractéristiques avancées, vous obtiendrez :

- 🔍 Un modèle utilisant toutes les statistiques disponibles
- 📊 Une évaluation plus robuste grâce à la validation croisée
- 💡 Des informations détaillées sur l'importance des caractéristiques
- 🎛️ Un contrôle plus fin des hyperparamètres des modèles

⚠️ Attention : utiliser toutes ces options ensemble peut ralentir considérablement le temps d'entraînement du modèle, surtout avec la validation croisée.

## 🧠 Algorithmes disponibles

### 🌲 Random Forest

La forêt aléatoire est un ensemble d'arbres de décision entraînés sur différents sous-ensembles des données. Chaque arbre "vote" pour une classe, et la classe ayant le plus de votes est choisie comme prédiction.

**Avantages :**

- ✅ Très performant pour les problèmes de classification
- ✅ Peu sensible au surapprentissage
- ✅ Fournit des mesures d'importance des caractéristiques
- ✅ Gère bien les données non linéaires

**Inconvénients :**

- ❌ Peut être plus lent que d'autres algorithmes
- ❌ Moins interprétable qu'un seul arbre de décision

### 🔬 SVM (Support Vector Machine)

Les SVM cherchent à trouver un hyperplan optimal qui sépare les différentes classes dans un espace de grande dimension. Pour les problèmes non linéaires, ils utilisent un "kernel trick" pour projeter les données dans un espace de plus grande dimension.

**Avantages :**

- ✅ Très efficace dans les espaces de grande dimension
- ✅ Versatile grâce aux différents noyaux (linéaire, polynomial, RBF)
- ✅ Économe en mémoire

**Inconvénients :**

- ❌ Sensible au choix des hyperparamètres
- ❌ Moins performant sur de très grands jeux de données
- ❌ Ne fournit pas directement des probabilités (mais notre implémentation utilise `probability=True`)

## ⚙️ Paramètres des modèles

### 🔒 Force de régularisation

La force de régularisation contrôle le compromis entre ajustement aux données d'entraînement et généralisation :

- 🔼 Une valeur plus élevée = plus de régularisation = modèle plus simple = moins de surapprentissage
- 🔽 Une valeur plus faible = moins de régularisation = modèle plus complexe = risque de surapprentissage

Pour SVM : une régularisation plus forte (valeur plus élevée) diminue la valeur du paramètre C, ce qui crée une marge plus large et tolère plus d'erreurs.

Pour Random Forest : la régularisation est principalement contrôlée par la profondeur maximale des arbres.

### 📏 Profondeur max des arbres (Random Forest)

Ce paramètre limite la profondeur maximale de chaque arbre dans la forêt :

- 🔽 Une valeur plus faible (ex: 3-5) = arbres plus simples = moins de surapprentissage
- 🔼 Une valeur plus élevée (ex: 10-20) = arbres plus complexes = risque de surapprentissage

Limiter la profondeur des arbres est particulièrement important pour les petits jeux de données ou lorsque les caractéristiques sont bruitées.

## 📈 Performances des modèles

Après tests approfondis, voici les performances observées:

| Modèle        | Précision | Paramètres optimaux              | Remarques            |
| ------------- | --------- | -------------------------------- | -------------------- |
| Random Forest | 0.81      | regularization=10.0, max_depth=3 | Performance réaliste |
| SVM           | 0.78      | regularization=10.0              | Performance réaliste |

## 📁 Structure du projet

- 📱 `streamlit_app.py` - Fichier principal de l'application Streamlit
- 📊 `data_loader.py` - Module pour charger et traiter les données
- 🧠 `model.py` - Module pour les modèles d'apprentissage automatique
- 🔮 `prediction.py` - Module pour les fonctions de prédiction
- 🎨 `styles.py` - Module pour le CSS et les styles de l'application
- 📂 `data/` - Dossier contenant les fichiers CSV des saisons de football

## 💻 Installation

1. Clonez ce dépôt
2. Installez les dépendances:

```bash
pip install -r requirements.txt
```

## 🚀 Exécution de l'application

```bash
streamlit run streamlit_app.py
```

## 📦 Dépendances principales

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
