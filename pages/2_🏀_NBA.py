import streamlit as st
import numpy as np
import pandas as pd

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

# Create dummy dataframe
dummy_df = create_dummy_dataframe()

# Apply styling
styled_df = dummy_df.style.set_table_styles(
    {
        'Matchup': [
            {'selector': 'td', 'props': 'font-weight: bold; color: #ffffff; background-color: #007acc;'},
        ],
        'Home Pitcher': [
            {'selector': 'td', 'props': 'background-color: #e6f7ff;'},
        ],
        'Away Pitcher': [
            {'selector': 'td', 'props': 'background-color: #e6f7ff;'},
        ],
        'Predicted Winner': [
            {'selector': 'td', 'props': 'background-color: #f4e542; color: #000000; font-weight: bold;'},
        ],
        'Winner Odds': [
            {'selector': 'td', 'props': 'background-color: #e6ffcc; color: #006600; font-weight: bold;'},
        ],
    }
).set_properties(**{'text-align': 'center'})

# Convert the styled dataframe to HTML
styled_html = styled_df.render()

# Streamlit app
st.title('MLB Predictions - Dummy DataFrame')
st.write("This is a dummy dataframe to play around with the formatting.")
st.markdown(styled_html, unsafe_allow_html=True)
