import streamlit as st
import requests, pandas as pd, time, numpy as np, pickle, re, json
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import altair as alt


# Add sidebar with additional information or navigation
st.sidebar.header('About')
st.sidebar.write("""
    This application provides predictions for today's NBA games based on historical data and machine learning models. 
    The predictions include the expected winner, starting pitchers, and the odds for each game.
""")
st.markdown("""
    <style>
    .coming-soon-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 80vh;
        background: linear-gradient(135deg, #f3ec78, #af4261);
        color: black;
        font-family: 'Arial', sans-serif;
        text-align: center;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .coming-soon-text {
        font-size: 4em;
        font-weight: bold;
    }
    .coming-soon-subtext {
        font-size: 1.5em;
        margin-top: 20px;
    }
    .emoji {
        font-size: 2em;
        margin: 0 10px;
    }
    </style>
    <div class="coming-soon-container">
        <div>
            <div class="emoji">🚧</div>
            <div class="coming-soon-text">Coming Soon</div>
            <div class="emoji">🚧</div>
        </div>
    </div>
""", unsafe_allow_html=True)
# Add footer with additional links or information
st.markdown("""
    <div class="footer">
        <p>&copy; 2024 Cobb's ML Predictions. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
