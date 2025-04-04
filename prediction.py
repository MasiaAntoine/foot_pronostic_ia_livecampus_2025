import pandas as pd
import streamlit as st
import numpy as np

def get_team_stats(df, team, is_home=True, last_n_matches=5):
    """
    Extrait les statistiques d'une équipe à partir du dataset.
    """
    team_home_matches = df[df['HomeTeam'] == team].copy()
    team_away_matches = df[df['AwayTeam'] == team].copy()
    
    if 'Date' in df.columns:
        team_home_matches = team_home_matches.sort_values('Date', ascending=False)
        team_away_matches = team_away_matches.sort_values('Date', ascending=False)
    
    home_stats = {
        'matches_played': len(team_home_matches),
        'wins': sum(team_home_matches['FTR'] == 'H'),
        'draws': sum(team_home_matches['FTR'] == 'D'),
        'losses': sum(team_home_matches['FTR'] == 'A'),
        'goals_scored': team_home_matches['FTHG'].sum() if 'FTHG' in team_home_matches.columns else 0,
        'goals_conceded': team_home_matches['FTAG'].sum() if 'FTAG' in team_home_matches.columns else 0
    }
    
    away_stats = {
        'matches_played': len(team_away_matches),
        'wins': sum(team_away_matches['FTR'] == 'A'),
        'draws': sum(team_away_matches['FTR'] == 'D'),
        'losses': sum(team_away_matches['FTR'] == 'H'),
        'goals_scored': team_away_matches['FTAG'].sum() if 'FTAG' in team_away_matches.columns else 0,
        'goals_conceded': team_away_matches['FTHG'].sum() if 'FTHG' in team_away_matches.columns else 0
    }
    
    recent_matches = pd.concat([
        team_home_matches.head(last_n_matches),
        team_away_matches.head(last_n_matches)
    ])
    if 'Date' in recent_matches.columns:
        recent_matches = recent_matches.sort_values('Date', ascending=False).head(last_n_matches)
    
    recent_form = []
    for _, match in recent_matches.iterrows():
        if match['HomeTeam'] == team:
            if match['FTR'] == 'H':
                recent_form.append(3)  
            elif match['FTR'] == 'D':
                recent_form.append(1) 
            else:
                recent_form.append(0)
        else:
            if match['FTR'] == 'A':
                recent_form.append(3)  
            elif match['FTR'] == 'D':
                recent_form.append(1) 
            else:
                recent_form.append(0)  
    
    avg_form = sum(recent_form) / len(recent_form) if recent_form else 1.5
    
    return home_stats if is_home else away_stats, avg_form

def get_head_to_head(df, home_team, away_team, last_n_matches=5):
    """
    Extrait les statistiques des confrontations directes entre deux équipes.
    """
    h2h_matches = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
                     ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))].copy()
    
    if 'Date' in h2h_matches.columns:
        h2h_matches = h2h_matches.sort_values('Date', ascending=False)
    
    h2h_matches = h2h_matches.head(last_n_matches)
    
    home_wins = 0
    away_wins = 0
    draws = 0
    
    for _, match in h2h_matches.iterrows():
        if match['HomeTeam'] == home_team and match['FTR'] == 'H':
            home_wins += 1
        elif match['AwayTeam'] == home_team and match['FTR'] == 'A':
            home_wins += 1
        elif match['HomeTeam'] == away_team and match['FTR'] == 'H':
            away_wins += 1
        elif match['AwayTeam'] == away_team and match['FTR'] == 'A':
            away_wins += 1
        else:
            draws += 1
    
    total_matches = len(h2h_matches)
    h2h_stats = {
        'total_matches': total_matches,
        'home_team_wins': home_wins,
        'away_team_wins': away_wins,
        'draws': draws,
        'home_win_rate': home_wins / total_matches if total_matches > 0 else 0.33,
        'away_win_rate': away_wins / total_matches if total_matches > 0 else 0.33,
        'draw_rate': draws / total_matches if total_matches > 0 else 0.33
    }
    
    return h2h_stats, h2h_matches

def predict_with_teams(df, home_team, away_team):
    """
    Prédit le résultat d'un match entre deux équipes en se basant sur leurs statistiques.
    """
    try:
        home_stats, home_form = get_team_stats(df, home_team, is_home=True)
        away_stats, away_form = get_team_stats(df, away_team, is_home=False)
        
        h2h_stats, h2h_matches = get_head_to_head(df, home_team, away_team)
        
        if home_stats is None or away_stats is None:
            raise ValueError("Statistiques d'équipe non disponibles")
            
        home_strength = 0.45
        away_strength = 0.25
        home_current_form = 1.5
        away_current_form = 1.5
        h2h_advantage = 0.5
        
        if home_stats['matches_played'] > 0:
            home_strength = home_stats['wins'] / home_stats['matches_played']
        
        if away_stats['matches_played'] > 0:
            away_strength = away_stats['wins'] / away_stats['matches_played']
        
        home_current_form = home_form if home_form is not None else 1.5
        away_current_form = away_form if away_form is not None else 1.5
        
        if h2h_stats['total_matches'] > 0:
            h2h_advantage = (h2h_stats['home_team_wins'] - h2h_stats['away_team_wins']) / h2h_stats['total_matches'] + 0.5
        
        home_win_prob = 0.45 
        draw_prob = 0.25
        away_win_prob = 0.30
        
        home_win_prob = home_win_prob * (1 + home_strength - 0.45) * (1 + home_current_form/3 - 0.5)
        away_win_prob = away_win_prob * (1 + away_strength - 0.25) * (1 + away_current_form/3 - 0.5)
        
        home_win_prob = home_win_prob * (h2h_advantage * 1.2)
        away_win_prob = away_win_prob * ((1 - h2h_advantage) * 1.2)
        
        total_prob = home_win_prob + draw_prob + away_win_prob
        home_win_prob = home_win_prob / total_prob
        draw_prob = draw_prob / total_prob
        away_win_prob = away_win_prob / total_prob
        
        probs = [home_win_prob, draw_prob, away_win_prob]
        predicted_result = ['H', 'D', 'A'][probs.index(max(probs))]
        
        return predicted_result, [home_win_prob, draw_prob, away_win_prob], {
            'home_stats': home_stats,
            'away_stats': away_stats,
            'h2h_stats': h2h_stats,
            'home_form': home_form,
            'away_form': away_form
        }
    except Exception as e:
        st.error(f"Erreur dans la prédiction: {str(e)}")
        # Retourner des valeurs par défaut en cas d'erreur
        default_probs = [0.4, 0.3, 0.3]
        return 'H', default_probs, {
            'home_stats': {'matches_played': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'goals_scored': 0, 'goals_conceded': 0},
            'away_stats': {'matches_played': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'goals_scored': 0, 'goals_conceded': 0},
            'h2h_stats': {'total_matches': 0, 'home_team_wins': 0, 'away_team_wins': 0, 'draws': 0, 
                        'home_win_rate': 0.33, 'away_win_rate': 0.33, 'draw_rate': 0.33},
            'home_form': 1.5,
            'away_form': 1.5
        }
