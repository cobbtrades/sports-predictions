import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="NBA Predictions", layout="wide")

st.title("NBA Predictions")
st.markdown("""
Here are the latest predictions for NBA games.
""")

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
  
if st.button('Generate Predictions'):
    with st.spinner('Generating predictions...'):
        final_display_df = create_dummy_dataframe()
        styled_df = final_display_df.style.set_table_styles(
            {
                'Matchup': [
                    {'selector': '', 'props': 'font-weight: bold; color: #ffffff; background-color: #007acc;'},
                ],
                'Home Pitcher': [
                    {'selector': '', 'props': 'background-color: #e6f7ff;'},
                ],
                'Away Pitcher': [
                    {'selector': '', 'props': 'background-color: #e6f7ff;'},
                ],
                'Predicted Winner': [
                    {'selector': '', 'props': 'background-color: #f4e542; color: #000000; font-weight: bold;'},
                ],
                'Winner Odds': [
                    {'selector': '', 'props': 'background-color: #e6ffcc; color: #006600; font-weight: bold;'},
                ],
            }
        ).set_properties(**{'text-align': 'center'})
        st.dataframe(styled_df)
