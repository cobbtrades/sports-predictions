import streamlit as st

st.set_page_config(page_title="NFL Predictions", layout="wide")

st.title("NFL Predictions")
st.markdown("""
Here are the latest predictions for NFL games.
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
            <div class="coming-soon-subtext">Stay tuned for upcoming earnings!</div>
        </div>
    </div>
""", unsafe_allow_html=True)
