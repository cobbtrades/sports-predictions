import streamlit as st

st.set_page_config(page_title="Sports Predictor", layout="wide")

hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title("Welcome to Sports Predictor")
st.markdown("""
This app provides predictions for MLB, NBA, and NFL games. Use the navigation on the left to select the sport you are interested in.
""")
