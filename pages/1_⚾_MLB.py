import streamlit as st
import altair as alt
from streamlit_extras.badges import badge
from pages.mlb_helpers import fetch_and_process_batting_data, fetch_and_process_pitching_data, fetch_fanduel_mlb_odds, scrape_games, generate_predictions

st.set_page_config(
    page_title="Cobb's ML Predictions",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: #ffffff;
            font-family: 'Arial', sans-serif;
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
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #ffaf42;
            color: #000000;
            font-weight: bold;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #ffcf72;
            color: #000000;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #333333;
            color: white;
            text-align: center;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

def buymeacoffee():
    badge(type="buymeacoffee", name="cobbtradesg")

st.header('Welcome to the MLB Predictions Page')
st.subheader('Generate Predictions for Today\'s Games')

if 'predictions' not in st.session_state:
    st.session_state.predictions = None

if st.button('Generate Predictions'):
    with st.spinner('Generating predictions...'):
        final_display_df, error = generate_predictions()
        if error:
            st.error(f"Error generating predictions: {error}")
        else:
            st.session_state.predictions = final_display_df

if st.session_state.predictions is not None:
    st.markdown("### Today's Game Predictions")
    
    # Interactive Chart Example using Altair
    chart = alt.Chart(st.session_state.predictions).mark_bar().encode(
        x='Matchup',
        y='Winner Odds',
        color='Predicted Winner'
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)
    
    styled_df = st.session_state.predictions.style.set_table_styles(
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
    st.markdown(styled_html, unsafe_allow_html=True)

# Add sidebar with additional information or navigation
st.sidebar.header('About')
st.sidebar.write("""
    This application provides predictions for today's MLB games based on historical data and machine learning models. 
    The predictions include the expected winner, starting pitchers, and the odds for each game.
""")

with st.sidebar:
    buymeacoffee()

# Add footer with additional links or information
st.markdown("""
    <div class="footer">
        <p>&copy; 2024 Cobb's ML MLB Predictions. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
