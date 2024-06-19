import streamlit as st
from streamlit_extras.badges import badge

st.set_page_config(page_title="NFL Predictions", layout="wide")

st.title("NBA Predictions")
def buymeacoffee():
    badge(type="buymeacoffee", name="cobbtradesg")
st.markdown("""
Here are the latest predictions for NBA games.
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

# Add sidebar with additional information or navigation
st.sidebar.header('About')
st.sidebar.write("""
    This application provides predictions for today's NBA games based on historical data and machine learning models. 
    The predictions include the expected winner, and the odds for each game.
""")

with st.sidebar:
    buymeacoffee()

# Add footer with additional links or information
st.markdown("""
    <div class="footer">
        <p>&copy; 2024 Cobb's NBA ML Predictions. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
