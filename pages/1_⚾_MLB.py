import requests, pandas as pd, time, numpy as np, pickle, re, json, streamlit as st, altair as alt, base64
from datetime import datetime, timedelta
from streamlit_extras.badges import badge

st.set_page_config(page_title="Cobb's ML Predictions", page_icon="💰", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
        body {background-color: #1e1e1e; color: #ffffff; font-family: 'Arial', sans-serif;}
        .main .block-container {padding-top: 2rem;}
        .css-1d391kg p {font-size: 16px; color: #ffffff;}
        .css-145kmo2 {background-color: #333333; border: 1px solid #333333; color: #ffffff;}
        .css-1avcm0n .css-vy48ge {background-color: #2e2e2e; border: 1px solid #444444;}
        .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {color: #ffaf42; font-family: 'Arial', sans-serif;}
        .stButton>button {background-color: #ffaf42; color: #000000; font-weight: bold; border-radius: 10px;}
        .stButton>button:hover {background-color: #ffcf72; color: #000000;}
        .footer {position: fixed; left: 0; bottom: 0; width: 100%; background-color: #333333; color: white; text-align: center; padding: 10px;}
    </style>
""", unsafe_allow_html=True)

def buymeacoffee():
    badge(type="buymeacoffee", name="cobbtradesg")

sport_dict = {"MLB": "mlb-baseball"}
teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']
tmap = {'KC': 'KCR', 'SD': 'SDP', 'SF': 'SFG', 'TB': 'TBR', 'WAS': 'WSN'}
team_name_mapping = {
    'ARI': 'Arizona Diamondbacks', 'ATL': 'Atlanta Braves', 'AZ': 'Arizona Diamondbacks', 'BAL': 'Baltimore Orioles', 'BOS': 'Boston Red Sox', 'CHC': 'Chicago Cubs', 'CHW': 'Chicago White Sox',
    'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians', 'COL': 'Colorado Rockies', 'DET': 'Detroit Tigers', 'HOU': 'Houston Astros', 'KCR': 'Kansas City Royals',
    'LAA': 'Los Angeles Angels', 'LAD': 'Los Angeles Dodgers', 'MIA': 'Miami Marlins', 'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins', 'NYM': 'New York Mets', 'NYY': 'New York Yankees',
    'OAK': 'Oakland Athletics', 'PHI': 'Philadelphia Phillies', 'PIT': 'Pittsburgh Pirates', 'SDP': 'San Diego Padres', 'SEA': 'Seattle Mariners', 'SFG': 'San Francisco Giants',
    'STL': 'St. Louis Cardinals', 'TBR': 'Tampa Bay Rays', 'TEX': 'Texas Rangers', 'TOR': 'Toronto Blue Jays', 'WSN': 'Washington Nationals'
}
team_acronyms = {v: k for k, v in team_name_mapping.items()}

@st.cache_data
def fetch_and_process_batting_data(team, year):
    try:
        batting_url = f'https://www.baseball-reference.com/teams/tgl.cgi?team={team}&t=b&year={year}'
        bat_df = pd.read_html(batting_url)[0]
        bat_df.insert(loc=0, column='Year', value=year)
        numeric_cols = ['PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'IBB', 'SO', 'HBP', 'SH', 'SF', 'ROE', 'GDP', 'SB', 'CS', 'LOB', 'BA', 'OBP', 'SLG', 'OPS']
        for col in numeric_cols:
            bat_df[col] = pd.to_numeric(bat_df[col], errors='coerce')
        bat_df['Rslt'] = bat_df['Rslt'].str.startswith('W').astype(int)
        bat_df = bat_df[bat_df['Opp'] != 'Opp']
        bat_df = bat_df[~bat_df['Date'].str.contains('susp', na=False)]
        bat_df['Date'] = bat_df['Date'].str.replace(r'\s+\(\d\)', '', regex=True)
        bat_df['Date'] = bat_df['Date'] + ', ' + bat_df['Year'].astype(str)
        bat_df['Date'] = pd.to_datetime(bat_df['Date'], format='%b %d, %Y')
        bat_df.rename(columns={'Unnamed: 3': 'Home', 'Opp. Starter (GmeSc)': 'OppStart'}, inplace=True)
        bat_df.insert(bat_df.columns.get_loc('Opp'), 'Tm', team)
        bat_df.drop(['Rk', 'Year', '#'], axis=1, inplace=True)
        bat_df['Home'] = bat_df['Home'].isna()
        bat_df['OppStart'] = bat_df['OppStart'].str.replace(r'\(.*\)', '', regex=True)
        return bat_df
    except Exception as e:
        st.error(f"Error fetching batting data for team {team}: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_and_process_pitching_data(team, year):
    try:
        pitching_url = f'https://www.baseball-reference.com/teams/tgl.cgi?team={team}&t=p&year={year}'
        pit_df = pd.read_html(pitching_url)[1]
        pit_df.insert(loc=0, column='Year', value=year)
        pit_df.insert(pit_df.columns.get_loc('Opp'), 'Tm', team)
        numeric_cols = ['H', 'R', 'ER', 'UER', 'BB', 'SO', 'HR', 'HBP', 'ERA', 'Pit', 'Str', 'IR', 'IS', 'SB', 'CS', 'AB', '2B', '3B', 'IBB', 'SH', 'SF', 'ROE', 'GDP']
        for col in numeric_cols:
            pit_df[col] = pd.to_numeric(pit_df[col], errors='coerce')
        pit_df = pit_df[pit_df['Opp'] != 'Opp']
        pit_df = pit_df[~pit_df['Date'].str.contains('susp', na=False)]
        pit_df['Date'] = pit_df['Date'].str.replace(r'\(\d\)', '', regex=True)
        pit_df['Date'] = pit_df['Date'] + ', ' + pit_df['Year'].astype(str)
        pit_df['Date'] = pd.to_datetime(pit_df['Date'], format='%b %d, %Y')
        pit_df.rename(columns={pit_df.columns[-1]: 'TmStart'}, inplace=True)
        pit_df['TmStart'] = pit_df['TmStart'].str.split().str[0]
        pit_df.drop(['Year', 'Rk', 'Unnamed: 3', 'Rslt', 'IP', 'BF', '#', 'Umpire'], axis=1, inplace=True)
        pit_pre = [col for col in pit_df.columns if col not in ['Gtm', 'Date', 'Tm', 'Opp', 'TmStart']]
        pit_df.rename(columns={col: f'p{col}' for col in pit_pre}, inplace=True)
        return pit_df
    except Exception as e:
        st.error(f"Error fetching pitching data for team {team}: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_fanduel_mlb_odds(date):
    try:
        sport = "MLB"
        date_str = date.strftime("%Y-%m-%d")
        spread_url = f"https://www.sportsbookreview.com/betting-odds/{sport_dict[sport]}/?date={date_str}"
        r = requests.get(spread_url)
        j = re.findall('__NEXT_DATA__" type="application/json">(.*?)</script>', r.text)
        if not j:
            return []
        build_id = json.loads(j[0])['buildId']
        odds_url = f"https://www.sportsbookreview.com/_next/data/{build_id}/betting-odds/{sport_dict[sport]}/money-line/full-game.json?league={sport_dict[sport]}&oddsType=money-line&oddsScope=full-game&date={date_str}"
        odds_json = requests.get(odds_url).json()
        if 'oddsTables' in odds_json['pageProps'] and odds_json['pageProps']['oddsTables']:
            games_list = odds_json['pageProps']['oddsTables'][0]['oddsTableModel']['gameRows']
        else:
            return []
        fanduel_games = []
        for game in games_list:
            if game['oddsViews']:
                for odds_view in game['oddsViews']:
                    if odds_view and odds_view.get('sportsbook', '').lower() == 'fanduel':
                        game_data = {
                            'date': game['gameView']['startDate'],
                            'home_team_abbr': game['gameView']['homeTeam']['shortName'],
                            'away_team_abbr': game['gameView']['awayTeam']['shortName'],
                            'home_ml': odds_view['currentLine'].get('homeOdds', 'N/A'),
                            'away_ml': odds_view['currentLine'].get('awayOdds', 'N/A')
                        }
                        fanduel_games.append(game_data)
                        break
        return fanduel_games
    except Exception as e:
        st.error(f"Error fetching Fanduel odds: {e}")
        return []

@st.cache_data
def scrape_games():
    try:
        date = datetime.today().strftime("%Y-%m-%d")
        spread_url = f"https://www.sportsbookreview.com/betting-odds/mlb-baseball/?date={date}"
        r = requests.get(spread_url)
        build_id = json.loads(re.findall('__NEXT_DATA__" type="application/json">(.*?)</script>', r.text)[0])['buildId']
        moneyline_url = f"https://www.sportsbookreview.com/_next/data/{build_id}/betting-odds/mlb-baseball/money-line/full-game.json?league=mlb-baseball&oddsType=money-line&oddsScope=full-game&date={date}"
        moneyline_json = requests.get(moneyline_url).json()
        games = []
        for g in moneyline_json['pageProps']['oddsTables'][0]['oddsTableModel']['gameRows']:
            game = {
                'date': g['gameView']['startDate'],
                'home_team_abbr': g['gameView']['homeTeam']['shortName'],
                'away_team_abbr': g['gameView']['awayTeam']['shortName'],
                'home_ml': None,
                'away_ml': None,
                'TmStart': 'TBD',
                'OppStart': 'TBD'
            }
            if 'homeStarter' in g['gameView'] and g['gameView']['homeStarter']:
                game['TmStart'] = g['gameView']['homeStarter'].get('firstInital', '') + "." + g['gameView']['homeStarter'].get('lastName', '')
            if 'awayStarter' in g['gameView'] and g['gameView']['awayStarter']:
                game['OppStart'] = g['gameView']['awayStarter'].get('firstInital', '') + "." + g['gameView']['awayStarter'].get('lastName', '')
            for line in g['oddsViews']:
                if line:
                    game['home_ml'] = line['currentLine']['homeOdds']
                    game['away_ml'] = line['currentLine']['awayOdds']
                    break
            games.append(game)
        return games, None
    except Exception as e:
        return [], str(e)

def generate_predictions():
    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    bat_data = []
    for team in teams:
        df = fetch_and_process_batting_data(team, 2024)
        bat_data.append(df)
        time.sleep(3)
    batting_df = pd.concat(bat_data, ignore_index=True)

    pit_data = []
    for team in teams:
        df = fetch_and_process_pitching_data(team, 2024)
        pit_data.append(df)
        time.sleep(3)
    pitching_df = pd.concat(pit_data, ignore_index=True)

    df2 = pd.merge(batting_df, pitching_df, on=['Gtm', 'Date', 'Tm', 'Opp'])

    start_date = datetime(2024, 3, 28).date()
    end_date = datetime.now().date()
    all_games = []
    for single_date in daterange(start_date, end_date):
        fanduel_games = fetch_fanduel_mlb_odds(single_date)
        all_games.extend(fanduel_games)

    odds2 = pd.DataFrame(all_games)
    odds2['home_team_abbr'] = odds2['home_team_abbr'].replace(tmap)
    odds2['away_team_abbr'] = odds2['away_team_abbr'].replace(tmap)
    odds2['date'] = pd.to_datetime(odds2['date']).dt.date

    df1 = pd.read_pickle('data/mlbgamelogs22-23.pkl')
    df_full = pd.concat([df1, df2])

    odds1 = pd.read_csv('data/mlbodds22-23.csv')[['date', 'home_team_abbr', 'away_team_abbr', 'home_ml', 'away_ml']]
    odds1['date'] = pd.to_datetime(odds1['date']).dt.date
    odds_full = pd.concat([odds1, odds2])
    odds_full['date'] = pd.to_datetime(odds_full['date'])

    odds1_home = odds_full.rename(columns={'home_ml': 'Tm_ml', 'away_ml': 'Opp_ml'})
    odds1_away = odds_full.rename(columns={'home_ml': 'Opp_ml', 'away_ml': 'Tm_ml'})

    merged_home = pd.merge(df_full[df_full['Home']], odds1_home, left_on=['Date', 'Tm'], right_on=['date', 'home_team_abbr'], how='left')
    merged_away = pd.merge(df_full[~df_full['Home']], odds1_away, left_on=['Date', 'Tm'], right_on=['date', 'away_team_abbr'], how='left')
    df_merged = pd.concat([merged_home, merged_away], ignore_index=True)
    df_merged = df_merged.drop(columns=['date', 'home_team_abbr', 'away_team_abbr'])
    df_merged = df_merged.dropna()

    df = df_merged.copy()
    df = df.sort_values(by=['Date', 'Tm'])

    stats_columns = ['R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'BA', 'OBP', 'pR', 'pH', 'p2B', 'p3B', 'pHR', 'pBB', 'pERA']
    for col in stats_columns:
        df[f'cumsum_{col}'] = df.groupby('Tm')[col].cumsum() - df[col]
        df[f'cumcount_{col}'] = df.groupby('Tm')[col].cumcount()

    for col in stats_columns:
        df[f'avg_{col}'] = df[f'cumsum_{col}'] / df[f'cumcount_{col}']

    df.drop(columns=[f'cumsum_{col}' for col in stats_columns] + [f'cumcount_{col}' for col in stats_columns], inplace=True)
    df.fillna(method='bfill', inplace=True)
    df['total'] = df['R'] + df['pR']

    with open('best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)

    games, error = scrape_games()
    if error:
        st.error(f"Error fetching games: {error}")
    else:
        data = {
            'Tm': [g['home_team_abbr'] for g in games],
            'Opp': [g['away_team_abbr'] for g in games],
            'TmStart': [g['TmStart'] for g in games],
            'OppStart': [g['OppStart'] for g in games],
            'Tm_ml': [g['home_ml'] for g in games],
            'Opp_ml': [g['away_ml'] for g in games]
        }
        todays_games = pd.DataFrame(data)
        todays_games.insert(loc=0, column='Home', value=True)
        todays_games['Tm'] = todays_games['Tm'].replace(tmap)
        todays_games['Opp'] = todays_games['Opp'].replace(tmap)
        todays_games['Date'] = pd.Timestamp('today').normalize()

    stats_columns = ['R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'BA', 'OBP', 'pR', 'pH', 'p2B', 'p3B', 'pHR', 'pBB', 'pERA']
    for col in stats_columns:
        todays_games[col] = 0

    df_combined = pd.concat([df, todays_games], ignore_index=True, sort=False)

    for col in stats_columns:
        df_combined[f'cumsum_{col}'] = df_combined.groupby('Tm')[col].cumsum() - df_combined[col]
        df_combined[f'cumcount_{col}'] = df_combined.groupby('Tm')[col].cumcount()

    for col in stats_columns:
        df_combined[f'avg_{col}'] = df_combined[f'cumsum_{col}'] / df_combined[f'cumcount_{col}']

    df_combined.fillna(method='ffill', inplace=True)
    df_combined.fillna(0, inplace=True)

    todays_games = df_combined[df_combined['Date'] == pd.Timestamp('today').normalize()]
    todays_games = todays_games[['Home', 'Tm', 'Opp', 'TmStart', 'OppStart', 'Tm_ml', 'Opp_ml'] + [f'avg_{col}' for col in stats_columns]]

    predicted_outcomes = best_model.predict(todays_games)
    predicted_probabilities = best_model.predict_proba(todays_games)
    predicted_confidence = np.max(predicted_probabilities, axis=1)
    predicted_outcomes_series = pd.Series(predicted_outcomes, index=todays_games.index, name='Predicted Outcome')
    todays_games['Predicted Outcome'] = predicted_outcomes_series
    todays_games['Predicted Outcome'] = todays_games['Predicted Outcome'].map({0: 'Loss', 1: 'Win'})
    todays_games['Tm'] = todays_games['Tm'].replace(team_name_mapping)
    todays_games['Opp'] = todays_games['Opp'].replace(team_name_mapping)
    todays_games['Predicted Winner'] = todays_games.apply(lambda row: row['Tm'] if row['Predicted Outcome'] == 'Win' else row['Opp'], axis=1)
    todays_games['Confidence'] = predicted_confidence
    todays_games['Confidence'] = (todays_games['Confidence'] * 100).round(2)
    todays_games.dropna(inplace=True)
    display_df = todays_games[['Tm', 'Opp', 'Tm_ml', 'Opp_ml', 'Predicted Winner', 'Confidence', 'TmStart', 'OppStart']].copy()
    display_df.rename(columns={'Tm': 'Home Team','Opp': 'Away Team','Tm_ml': 'Home Odds','Opp_ml': 'Away Odds','Predicted Winner': 'Predicted Winner','Confidence': 'Confidence','TmStart': 'Home Pitcher','OppStart': 'Away Pitcher'}, inplace=True)
    display_df['Losing Team'] = display_df.apply(lambda row: row['Away Team'] if row['Predicted Winner'] == row['Home Team'] else row['Home Team'], axis=1)
    display_df['Matchup'] = display_df.apply(lambda row: f"{row['Predicted Winner']} vs {row['Losing Team']}", axis=1)
    display_df['Winner Odds'] = display_df.apply(lambda row: row['Home Odds'] if row['Predicted Winner'] == row['Home Team'] else row['Away Odds'], axis=1)
    display_df['Winner Odds'] = display_df['Winner Odds'].astype(float).astype(int)
    final_display_columns = ['Matchup', 'Home Pitcher', 'Away Pitcher', 'Predicted Winner', 'Winner Odds']
    final_display_df = display_df[final_display_columns]
    return final_display_df, todays_games, df

st.header('Welcome to the MLB Predictions Page')
st.subheader('Generate Predictions for Today\'s Games')

def get_highest_confidence_game(todays_games):
    max_confidence_idx = todays_games['Confidence'].idxmax()
    return todays_games.loc[max_confidence_idx]

def get_historical_record(df, team1, team2):
    matchups = df[((df['Tm'] == team1) & (df['Opp'] == team2)) | ((df['Tm'] == team2) & (df['Opp'] == team1))]
    team1_wins = len(matchups[(matchups['Tm'] == team1) & (matchups['R'] > matchups['pR'])]) + len(matchups[(matchups['Opp'] == team1) & (matchups['R'] < matchups['pR'])])
    team2_wins = len(matchups[(matchups['Tm'] == team2) & (matchups['R'] > matchups['pR'])]) + len(matchups[(matchups['Opp'] == team2) & (matchups['R'] < matchups['pR'])])
    return team1_wins, team2_wins

if 'predictions' not in st.session_state:
    st.session_state.predictions = None

if st.button('Generate Predictions'):
    with st.spinner('Generating predictions...'):
        final_display_df, todays_games, df = generate_predictions()
        st.session_state.predictions = final_display_df
        st.session_state.todaygames = todays_games
        st.session_state.df = df

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
    
    styled_html = styled_df.to_html()

    lc, rc = st.columns([3,1])
    with lc:
        st.markdown("### Today's Game Predictions")
        st.markdown(styled_html, unsafe_allow_html=True)
    with rc:
        st.markdown(<div style="text-align: center;">"### Highest Confidence Prediction",</div>)
        highest_confidence_game = get_highest_confidence_game(st.session_state.todaygames)
        
        home_team = highest_confidence_game['Tm']
        away_team = highest_confidence_game['Opp']
        confidence = highest_confidence_game['Confidence']
        predicted_winner = highest_confidence_game['Predicted Winner']
        losing_team = away_team if predicted_winner == home_team else home_team
        winner_odds = highest_confidence_game['Tm_ml'] if predicted_winner == home_team else highest_confidence_game['Opp_ml']
        
        winner_logo_path = f'logos/{team_acronyms[predicted_winner]}.svg'
        loser_logo_path = f'logos/{team_acronyms[losing_team]}.svg'

        try:
            def svg_to_base64(svg_file_path):
                with open(svg_file_path, "rb") as svg_file:
                    svg_bytes = svg_file.read()
                    return base64.b64encode(svg_bytes).decode()

            winner_logo_base64 = svg_to_base64(winner_logo_path.lower())
            loser_logo_base64 = svg_to_base64(loser_logo_path.lower())

            st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/svg+xml;base64,{winner_logo_base64}" alt="{predicted_winner} Logo" style="height: 100px;">
            </div>
            <div style="text-align: center;">
                <h4>{predicted_winner} is predicted to win with {confidence:.2f}% confidence</h4>
            </div>
            <div style="text-align: center; margin-top: 20px;">
                <img src="data:image/svg+xml;base64,{loser_logo_base64}" alt="{losing_team} Logo" style="height: 100px;">
            </div>
            """, unsafe_allow_html=True)

        except FileNotFoundError as e:
            st.error(f"Error loading team logos: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

        team1_wins, team2_wins = get_historical_record(st.session_state.df, team_acronyms[predicted_winner], team_acronyms[losing_team])
        st.markdown(f"""
        <div style="text-align: center;">
            <h4>Historical Record</h4>
            <h5>{team_acronyms[predicted_winner]} vs {team_acronyms[losing_team]}: {team1_wins} - {team2_wins}</h5>
        </div>
        """, unsafe_allow_html=True)

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
        <p>&copy; 2024 Cobb's MLB ML Predictions. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
