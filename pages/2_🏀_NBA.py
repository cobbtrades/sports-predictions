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

# Add footer with additional links or information
st.markdown("""
    <div class="footer">
        <p>&copy; 2024 Cobb's ML Predictions. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
