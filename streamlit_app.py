import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import des modules personnalisés
from data_loader import get_processed_data
from model import train_model
from prediction import predict_with_teams
from styles import get_css

# Configuration de la page
st.set_page_config(page_title="Prédiction Ligue 1", layout="wide")

# Application du CSS
st.markdown(get_css(), unsafe_allow_html=True)

# Titre principal
st.markdown("<h1 class='main-header'>Prédiction des Résultats de Matchs de Ligue 1</h1>", unsafe_allow_html=True)

# Barre latérale pour les options
st.sidebar.title("Options")

advanced_mode = st.sidebar.checkbox("Mode avancé", value=False)

use_advanced_features = st.sidebar.checkbox("Utiliser des caractéristiques avancées", value=True)
selected_model = st.sidebar.selectbox(
    "Choisir un algorithme",
    ["Random Forest", "SVM"]
)

st.sidebar.markdown("### Paramètres de régularisation")
st.sidebar.markdown("👉 _Une régularisation plus forte (valeurs plus élevées) réduit le surapprentissage_")

max_reg = 10.0
regularization = st.sidebar.slider("Force de régularisation", 0.01, max_reg, 10.0, 0.1)

max_depth = None
if selected_model == "Random Forest":
    max_depth = st.sidebar.slider(f"Profondeur max des arbres ({selected_model})", 
                                 3, 20, 3)

use_cv = False
n_folds = 5
if advanced_mode:
    st.sidebar.markdown("### Validation croisée")
    use_cv = st.sidebar.checkbox("Utiliser la validation croisée", value=False,
                               help="Évalue le modèle sur plusieurs partitions des données pour une estimation plus fiable")
    if use_cv:
        n_folds = st.sidebar.slider("Nombre de partitions", 3, 10, 5,
                                 help="Plus de partitions = évaluation plus robuste mais plus lente")

# Chargement et prétraitement des données
df, teams, default_features, advanced_features = get_processed_data()

with st.expander("Informations sur le dataset"):
    st.write(f"Nombre total de matchs: {len(df)}")
    st.write(f"Période couverte: {df['Date'].min().strftime('%d/%m/%Y')} - {df['Date'].max().strftime('%d/%m/%Y')}")
    
    if 'FTR' in df.columns:
        result_counts = df['FTR'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=result_counts.index, y=result_counts.values, ax=ax)
        ax.set_title('Distribution des résultats')
        ax.set_xlabel('Résultat')
        ax.set_ylabel('Nombre de matchs')
        st.pyplot(fig)

# Sélection des caractéristiques
if use_advanced_features and advanced_features:
    selected_features = default_features + advanced_features
    st.info(f"Utilisation de {len(selected_features)} caractéristiques pour le modèle")
else:
    selected_features = default_features
    st.info("Utilisation des cotes uniquement pour le modèle")

# Entraînement du modèle
model, scaler, label_encoder, accuracy, conf_matrix, class_report, X_test, y_test, feature_importance, cv_results = train_model(
    df, selected_features, selected_model, regularization, max_depth, use_cv, n_folds, advanced_mode
)

# Affichage des performances du modèle
st.markdown("<h2 class='sub-header'>Performance du modèle</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if cv_results:
        st.metric("Précision du modèle (test)", f"{accuracy:.2f}")
        st.metric("Précision validation croisée", f"{cv_results['mean']:.2f} ± {cv_results['std']:.2f}")
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(1, len(cv_results['scores'])+1), cv_results['scores'])
        ax.axhline(y=cv_results['mean'], color='r', linestyle='-', label=f"Moyenne: {cv_results['mean']:.2f}")
        ax.set_xlabel('Fold')
        ax.set_ylabel('Précision')
        ax.set_title('Résultats de la validation croisée')
        ax.legend()
        st.pyplot(fig)
    else:
        st.metric("Précision du modèle", f"{accuracy:.2f}")
    
    if accuracy > 0.95:
        st.warning("⚠️ La précision est anormalement élevée, ce qui suggère un surapprentissage. Le modèle ne sera probablement pas aussi performant sur de nouvelles données. Essayez d'augmenter la régularisation ou d'utiliser moins de caractéristiques.")
        
        if selected_model == "Régression Logistique":
            st.error("⚙️ La régression logistique montre un fort surapprentissage même avec régularisation=10. Essayez d'activer le mode avancé et utilisez le penalty 'elasticnet'.")
        elif selected_model == "Gradient Boosting":
            st.error("⚙️ Le Gradient Boosting montre un fort surapprentissage même avec regularization=10 et max_depth=3. Activez le mode avancé pour utiliser des techniques supplémentaires.")
    elif accuracy > 0.75:
        st.success("✅ Bonne précision ! Un modèle avec une précision autour de 0.75-0.85 est généralement fiable pour les prédictions sportives.")
    elif accuracy > 0.55:
        st.info("ℹ️ Précision modérée. Pour les prédictions sportives, c'est souvent un indicateur réaliste.")
    else:
        st.error("❌ Précision faible. Essayez d'autres combinaisons de caractéristiques ou d'autres algorithmes.")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Domicile (H)', 'Nul (D)', 'Extérieur (A)'],
                yticklabels=['Domicile (H)', 'Nul (D)', 'Extérieur (A)'], ax=ax)
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Valeur réelle')
    ax.set_title('Matrice de confusion')
    st.pyplot(fig)

with col2:
    st.write("Rapport de classification:")
    report_df = pd.DataFrame(class_report).transpose()
    st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))
    
    if feature_importance:
        st.write("Importance des caractéristiques:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        importance_df = pd.DataFrame(sorted_features, columns=['Caractéristique', 'Importance'])
        
        importance_df['Importance'] = importance_df['Importance'] / importance_df['Importance'].sum()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Caractéristique', data=importance_df, ax=ax)
        ax.set_title('Importance relative des caractéristiques')
        ax.set_xlabel('Importance normalisée')
        st.pyplot(fig)
        
        if len(importance_df) > 0:
            st.markdown(f"**Caractéristique la plus importante:** `{importance_df.iloc[0]['Caractéristique']}`")
            
            if 'B365' in importance_df.iloc[0]['Caractéristique']:
                st.markdown("_Les cotes des bookmakers sont généralement d'excellents prédicteurs car elles intègrent déjà beaucoup d'informations._")

# Section de prédiction avec les cotes
st.markdown("<h2 class='sub-header'>Prédiction pour un match</h2>", unsafe_allow_html=True)

if st.checkbox("Prédire avec les cotes"):
    col1, col2, col3 = st.columns(3)
    with col1:
        b365h = st.number_input("Cote Victoire Domicile", value=2.0, step=0.1)
    with col2:
        b365d = st.number_input("Cote Match Nul", value=3.0, step=0.1)
    with col3:
        b365a = st.number_input("Cote Victoire Extérieur", value=3.5, step=0.1)
    
    input_data = [b365h, b365d, b365a]
    feature_names = ['B365H', 'B365D', 'B365A']
    
    if use_advanced_features and advanced_features:
        for feature in advanced_features:
            val = st.slider(f"Valeur pour {feature}", 0, 20, 5)
            input_data.append(val)
            feature_names.append(feature)
    
    input_array = np.array([input_data])
    
    if st.button("Prédire le résultat (cotes)"):
        scaled_input = scaler.transform(input_array)
        
        prediction_proba = model.predict_proba(scaled_input)[0]
        prediction = model.predict(scaled_input)[0]
        result = label_encoder.inverse_transform([prediction])[0]
        
        result_texts = {
            'H': ("Victoire de l'équipe à domicile", "win"),
            'D': ("Match nul", "draw"),
            'A': ("Victoire de l'équipe à l'extérieur", "loss")
        }
        
        result_text, result_class = result_texts[result]
        
        st.markdown(f"<div class='result-box {result_class}'>{result_text} ({result})</div>", unsafe_allow_html=True)
        
        st.write("Probabilités prédites:")
        proba_df = pd.DataFrame({
            'Résultat': ['Victoire domicile (H)', 'Match nul (D)', 'Victoire extérieur (A)'],
            'Probabilité': prediction_proba
        })
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Résultat', y='Probabilité', data=proba_df, ax=ax)
        ax.set_ylim(0, 1)
        for i, p in enumerate(prediction_proba):
            ax.annotate(f'{p:.2f}', (i, p), ha='center', va='bottom')
        st.pyplot(fig)

# Section de prédiction avec les équipes
st.markdown("### Ou sélectionnez les équipes")
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Équipe à domicile", teams)
with col2:
    away_team = st.selectbox("Équipe à l'extérieur", teams, index=1 if len(teams) > 1 else 0)

if st.button("Prédire le résultat (équipes)"):
    try:
        with st.spinner("Analyse des performances des équipes..."):
            predicted_result, probabilities, stats = predict_with_teams(df, home_team, away_team)
            
            result_texts = {
                'H': (f"Victoire de {home_team}", "win"),
                'D': ("Match nul", "draw"),
                'A': (f"Victoire de {away_team}", "loss")
            }
            
            result_text, result_class = result_texts[predicted_result]
            
            st.markdown(f"<div class='result-box {result_class}'>{result_text} ({predicted_result})</div>", unsafe_allow_html=True)
            
            st.write("Probabilités prédites:")
            proba_df = pd.DataFrame({
                'Résultat': [f'Victoire {home_team} (H)', 'Match nul (D)', f'Victoire {away_team} (A)'],
                'Probabilité': probabilities
            })
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Résultat', y='Probabilité', data=proba_df, ax=ax)
            ax.set_ylim(0, 1)
            for i, p in enumerate(probabilities):
                ax.annotate(f'{p:.2f}', (i, p), ha='center', va='bottom')
            st.pyplot(fig)
            
            with st.expander("Détails des statistiques utilisées"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"Statistiques de {home_team}")
                    st.write(f"Matchs joués à domicile: {stats['home_stats']['matches_played']}")
                    st.write(f"Victoires: {stats['home_stats']['wins']}")
                    st.write(f"Nuls: {stats['home_stats']['draws']}")
                    st.write(f"Défaites: {stats['home_stats']['losses']}")
                    st.write(f"Forme récente (0-3): {stats['home_form']:.2f}")
                
                with col2:
                    st.subheader(f"Statistiques de {away_team}")
                    st.write(f"Matchs joués à l'extérieur: {stats['away_stats']['matches_played']}")
                    st.write(f"Victoires: {stats['away_stats']['wins']}")
                    st.write(f"Nuls: {stats['away_stats']['draws']}")
                    st.write(f"Défaites: {stats['away_stats']['losses']}")
                    st.write(f"Forme récente (0-3): {stats['away_form']:.2f}")
                
                st.subheader("Confrontations directes")
                st.write(f"Nombre de matchs: {stats['h2h_stats']['total_matches']}")
                st.write(f"Victoires {home_team}: {stats['h2h_stats']['home_team_wins']}")
                st.write(f"Victoires {away_team}: {stats['h2h_stats']['away_team_wins']}")
                st.write(f"Matchs nuls: {stats['h2h_stats']['draws']}")
    
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        st.info("Les données historiques pourraient être insuffisantes pour ces équipes. Essayez une autre paire d'équipes ou utilisez la prédiction basée sur les cotes.")

st.markdown("---")
st.markdown("Développé pour LiveCampus | 2025")
