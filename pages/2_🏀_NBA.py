import streamlit as st
import pandas as pd
import numpy as np

# Custom CSS for better styling
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .main .block-container {
            padding-top: 2rem;
        }
        .css-1d391kg p {
            font-size: 16px;
            color: #ffffff;
        }
        .css-145kmo2 {
            background-color: #333333;
            border: 1px solid #333333;
            color: #ffffff;
        }
        .css-1avcm0n .css-vy48ge {
            background-color: #2e2e2e;
            border: 1px solid #444444;
        }
        .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
            color: #ffaf42;
        }
    </style>
""", unsafe_allow_html=True)

# Function to create a dummy dataframe
def create_dummy_dataframe():
    data = {
        'Matchup': [
            f"Team {i} vs Team {j}" for i, j in zip(np.random.choice(range(1, 6), 10), np.random.choice(range(1, 6), 10))
        ],
        'Home Pitcher': [f"Pitcher {i}" for i in range(1, 11)],
        'Away Pitcher': [f"Pitcher {j}" for j in range(11, 21)],
        'Predicted Winner': [f"Team {i}" for i in np.random.choice(range(1, 6), 10)],
        'Winner Odds': np.random.randint(-200, 200, 10)
    }
    return pd.DataFrame(data)

st.title('MLB Predictions')
st.header('Welcome to the MLB Predictions Page')
st.subheader('Generate Predictions for Today\'s Games')


# Create dummy dataframe
dummy_df = create_dummy_dataframe()

# Apply styling
styled_df = dummy_df.style.set_table_styles(
    {
        'Matchup': [
            {'selector': 'td', 'props': 'font-weight: bold; color: #ffaf42; background-color: #000000;'},
        ],
        'Home Pitcher': [
            {'selector': 'td', 'props': 'font-weight: bold; color: #ffffff; background-color: #000000;'},
        ],
        'Away Pitcher': [
            {'selector': 'td', 'props': 'font-weight: bold; color: #ffffff; background-color: #000000;'},
        ],
        'Predicted Winner': [
            {'selector': 'td', 'props': 'background-color: #000000; color: #49f770; font-weight: bold;'},
        ],
        'Winner Odds': [
            {'selector': 'td', 'props': 'background-color: #000000; color: #2daefd; font-weight: bold;'},
        ],
    }
).set_properties(**{'text-align': 'center'}).hide(axis='index')

# Convert the styled dataframe to HTML
styled_html = styled_df.to_html()

# Streamlit app
st.title('MLB Predictions - Dummy DataFrame')
st.write("This is a dummy dataframe to play around with the formatting.")
st.markdown(styled_html, unsafe_allow_html=True)

st.sidebar.header('About')
st.sidebar.write("""
    This application provides predictions for today's MLB games based on historical data and machine learning models. 
    The predictions include the expected winner, starting pitchers, and the odds for each game.
""")
st.sidebar.subheader('Navigation')
st.sidebar.write("""
    - Home
    - Predictions
    - About
""")

# Add footer with additional links or information
st.markdown("""
    <div style="position: fixed; left: 0; bottom: 0; width: 100%; background-color: #333333; color: white; text-align: center; padding: 10px;">
        <p>&copy; 2024 MLB Predictions. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
