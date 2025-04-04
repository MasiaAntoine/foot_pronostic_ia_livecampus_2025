import pandas as pd
import streamlit as st

def load_data():
    """
    Charge les données des saisons de football depuis les fichiers CSV.
    """
    season_files = [
    'data/F1_2015_2016.csv', 'data/F1_2016_2017.csv', 'data/F1_2017_2018.csv',
    'data/F1_2018_2019.csv', 'data/F1_2019_2020.csv', 'data/F1_2020_2021.csv',
    'data/F1_2021_2022.csv', 'data/F1_2022_2023.csv'
    ]

    df_list = [pd.read_csv(file, parse_dates=['Date'], dayfirst=True) for file in season_files]
    df = pd.concat(df_list, ignore_index=True)
    return df

@st.cache_data
def get_processed_data():
    """
    Traite les données et extrait les informations nécessaires pour l'application.
    """
    df = load_data()
    
    st.write("Colonnes disponibles dans le dataset:", df.columns.tolist())
    
    teams = []
    try:
        if 'HomeTeam' in df.columns and 'AwayTeam' in df.columns:
            home_teams = [str(team) for team in df['HomeTeam'].unique() if not pd.isna(team)]
            away_teams = [str(team) for team in df['AwayTeam'].unique() if not pd.isna(team)]
            teams = sorted(list(set(home_teams) | set(away_teams)))
        elif 'Home' in df.columns and 'Away' in df.columns:
            home_teams = [str(team) for team in df['Home'].unique() if not pd.isna(team)]
            away_teams = [str(team) for team in df['Away'].unique() if not pd.isna(team)]
            teams = sorted(list(set(home_teams) | set(away_teams)))
        else:
            teams = ["Paris SG", "Marseille", "Lyon", "Lille", "Monaco", "Rennes", 
                     "Nice", "Lens", "Strasbourg", "Nantes", "Montpellier", "Reims"]
            st.warning("Colonnes des équipes non trouvées. Utilisation d'une liste prédéfinie.")
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des équipes: {e}")
        teams = ["Équipe A", "Équipe B", "Équipe C", "Équipe D"]
    
    st.write(f"Nombre d'équipes trouvées: {len(teams)}")
    if len(teams) < 10:
        st.write("Équipes:", teams)
    
    default_features = []
    for col in ['B365H', 'B365D', 'B365A']:
        if col in df.columns:
            default_features.append(col)
    
    if not default_features:
        st.warning("Colonnes de cotes non trouvées. Vérifiez le format de vos données.")
        df['B365H'] = 2.0
        df['B365D'] = 3.0 
        df['B365A'] = 3.5
        default_features = ['B365H', 'B365D', 'B365A']
    
    advanced_features = []
    potential_features = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    for col in potential_features:
        if col in df.columns:
            advanced_features.append(col)
    
    if 'FTR' not in df.columns:
        st.error("La colonne 'FTR' (résultat final) n'est pas présente dans le dataset!")
        df['FTR'] = 'H'
        
    return df, teams, default_features, advanced_features
