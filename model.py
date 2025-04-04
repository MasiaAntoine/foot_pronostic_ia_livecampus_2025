import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

@st.cache_resource
def train_model(df, selected_features, model_name, regularization_value, max_depth_value, use_cross_validation=False, cv_folds=5, advanced_mode=False):
    """
    Entraîne un modèle de prédiction à partir des caractéristiques sélectionnées.
    """
    model_df = df[selected_features + ['FTR']].dropna()
    
    label_encoder = LabelEncoder()
    model_df['FTR'] = label_encoder.fit_transform(model_df['FTR'])
    
    scaler = StandardScaler()
    model_df[selected_features] = scaler.fit_transform(model_df[selected_features])
    
    X = model_df[selected_features]
    y = model_df['FTR']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    if model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=max_depth_value, 
            min_samples_split=5,
            min_samples_leaf=3 if advanced_mode else 1,
            class_weight='balanced', 
            random_state=42
        )
    elif model_name == "SVM":
        model = SVC(
            C=1/regularization_value, 
            probability=True, 
            class_weight='balanced',
            kernel='rbf',
            gamma='scale'
        )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    cv_results = None
    if use_cross_validation:
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        cv_results = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
    
    feature_importance = None
    if model_name == "Random Forest":
        feature_importance = dict(zip(selected_features, model.feature_importances_))
    elif model_name == "SVM" and advanced_mode:
        r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        feature_importance = dict(zip(selected_features, r.importances_mean))
    
    return model, scaler, label_encoder, test_accuracy, conf_matrix, class_report, X_test, y_test, feature_importance, cv_results
