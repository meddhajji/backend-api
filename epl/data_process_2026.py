import pandas as pd
from rapidfuzz import process, utils
from fuzzywuzzy import process
import numpy as np
import os

def process_data():
    print("Processing data and updating Elo ratings...")
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    epl2025 = pd.read_csv(os.path.join(data_dir, "epl_2025.csv"))
    epl2526 = pd.read_csv(os.path.join(data_dir, "epl_2526.csv"))
    epl2026 = pd.concat([epl2025, epl2526], ignore_index=True)

    # ... (rest of function until end) ...
    
    final_df.to_csv(os.path.join(data_dir, "epl_2026.csv"), index=False)
    print("Saved epl_2026.csv")

    teams_2026 = pd.concat([
        epl2026['team'], epl2026['opponent'],
    ]).unique()

    teams = ['Fulham', 'Crystal Palace', 'Liverpool', 'West Ham United',
    'West Bromwich Albion', 'Tottenham Hotspur', 'Brighton',
    'Sheffield United', 'Everton', 'Leeds United', 'Manchester United', 'Arsenal',
    'Leicester City', 'Chelsea', 'Southampton', 'Newcastle United', 'Aston Villa',
    'Wolves', 'Burnley', 'Manchester City', 'Brentford',
    'Norwich City', 'Watford', 'Bournemouth', 'Nottingham Forest', 'Luton Town',
    'Ipswich Town', 'Sunderland', 'Wolverhampton Wanderers'] 


    mapping = {
        'Man United': 'Manchester United',
        'Man City': 'Manchester City',
        "Nott'm Forest": "Nottingham Forest",
        "Leicester": "Leicester City",
        "Newcastle": "Newcastle United",
    }

    # 5. The Loop
    for team in teams_2026:
        # Skip if already hardcoded
        if team in mapping:
            continue
            
        # Using rapidfuzz to find best match
        match, score = process.extractOne(team, teams)
        # If the exact name is already in the target list, map it to itself (identity)
        if team in teams:
            mapping[team] = team
        elif score >= 86:
            mapping[team] = match
        else:
            print(f"Unmapped: '{team}' (Best guess: {match}, Score: {score})")
            # if that's not printed means all teams are mapped 

    # Apply
    epl2026['team'] = epl2026['team'].replace(mapping)
    epl2026['opponent'] = epl2026['opponent'].replace(mapping)
    epl2026['pts'] = epl2026['result'].map({'W': 3, 'D': 1, 'L': 0})

    # Standardize Dates
    epl2026['date'] = pd.to_datetime(epl2026['date'])

    def create_match_id(row, home_col, away_col, date_col):
        # We use Season_Home_Away to avoid date-shift issues (postponed games)
        # season is needed, let's derive it if not present
        year = row[date_col].year
        season = year if row[date_col].month < 7 else year + 1
        return f"{season}_{row[home_col]}_{row[away_col]}"

    epl2026['match_id'] = epl2026.apply(lambda r: create_match_id(r, 'team', 'opponent', 'date'), axis=1)


    # 4. Calculate Dual Rolling Averages
    epl2026 = epl2026.sort_values(['team', 'date'])

    # A. Overall Form (Last 5 games anywhere)
    epl2026['roll_pts_5'] = epl2026.groupby('team')['pts'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    ).round(2)

    # B. Venue-Specific Form (Last 3 games at Home or Away)
    epl2026['venue_roll_pts_3'] = epl2026.groupby(['team', 'venue'])['pts'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    ).round(2)

    # Fill NaNs with median
    median_pts = epl2026['roll_pts_5'].median()
    epl2026['roll_pts_5'] = epl2026['roll_pts_5'].fillna(median_pts)
    median_venue_pts = epl2026['venue_roll_pts_3'].median()
    epl2026['venue_roll_pts_3'] = epl2026['venue_roll_pts_3'].fillna(median_venue_pts)

    # Ensure rolling stats are NaN if the previous game didn't happen
    valid_history_mask = epl2026.groupby('team')['result'].shift(1).notna()
    epl2026['season_int'] = epl2026['match_id'].apply(lambda x: int(x.split('_')[0]))

    # Wherever the mask is False (meaning the previous game didn't happen), force the rolling stat to NaN.
    epl2026.loc[~valid_history_mask & (epl2026['season_int'] == 2026), ['roll_pts_5', 'venue_roll_pts_3']] = np.nan


    # 6. Pivot from long dataset to wide (3800 rows -> 1900 rows)

    features = ['round', 'date', 'team', 'opponent', 'venue', 'result', 'gf', 'ga', 'roll_pts_5', 'venue_roll_pts_3']
    df_features = epl2026[features]

    home_feats = df_features[df_features['venue'] == 'Home'].copy()
    away_feats = df_features[df_features['venue'] == 'Away'].copy()

    # Add prefixes to distinguish
    home_feats = home_feats.rename(columns={c: 'home_'+c for c in ['roll_pts_5', 'venue_roll_pts_3']})
    away_feats = away_feats.rename(columns={c: 'away_'+c for c in ['roll_pts_5', 'venue_roll_pts_3']})

    # Final Match ID for Join
    home_feats['match_id'] = home_feats.apply(lambda r: create_match_id(r, 'team', 'opponent', 'date'), axis=1)
    away_feats['match_id'] = away_feats.apply(lambda r: create_match_id(r, 'opponent', 'team', 'date'), axis=1)

    # 7. Final Master Merge
    # Combine Home and Away features
    df_final = pd.merge(home_feats, away_feats.drop(['round', 'date', 'team', 'opponent', 'venue', 'result', 'gf', 'ga'], axis=1), on='match_id', how='inner')

    # the elo and odds calculations :
    df = df_final.sort_values('date').reset_index(drop=True)

    elo_dict = {team: 1500 for team in teams}
    K = 30
    HOME_ADV = 100

    def get_expected_score(h_elo, a_elo):
        return 1 / (10**(-(h_elo + HOME_ADV - a_elo) / 400) + 1)

    def get_mov_multiplier(h_goals, a_goals):
        diff = abs(h_goals - a_goals)
        if diff <= 1: return 1
        elif diff == 2: return 1.5
        else: return (11 + diff) / 8

    results = []
    current_season = df.iloc[0]['match_id'].split('_')[0]

    for index, row in df.iterrows():
        row_season = row['match_id'].split('_')[0]
        if row_season != current_season:
            for team in elo_dict:
                elo_dict[team] = round(elo_dict[team] + (1500 - elo_dict[team]) * 0.33, 2)
            current_season = row_season

        h_team, a_team = row['team'], row['opponent']
        row_data = row.to_dict()
        
        # 1. Elo Features
        row_data['home_elo_pre'] = elo_dict[h_team]
        row_data['away_elo_pre'] = elo_dict[a_team]
        row_data['elo_diff'] = round(elo_dict[h_team] - elo_dict[a_team], 2)
        
        # 2. Only update Elo if match has been played
        
        exp_h = get_expected_score(elo_dict[h_team], elo_dict[a_team])
        mov = get_mov_multiplier(row['gf'], row['ga'])
        
        #result_map = {'W': 1, 'D': 0.5, 'L': 0}
        #actual_h = result_map[row['result']]
        actual_h  = 1 if row['result'] == 'W' else (0.5 if row['result'] == 'D' else 0)

        
        exchange = K * mov * (actual_h - exp_h)
        elo_dict[h_team] = round(elo_dict[h_team] + exchange, 2)
        elo_dict[a_team] = round(elo_dict[a_team] - exchange, 2)
        
        results.append(row_data)
        
    final_df = pd.DataFrame(results)
    final_df = final_df[['round', 'date', 'team', 'opponent', 'venue', 'result', 'gf', 'ga', 'match_id', 'home_roll_pts_5', 'home_venue_roll_pts_3', 'away_roll_pts_5', 'away_venue_roll_pts_3', 'home_elo_pre', 'away_elo_pre', 'elo_diff']]
    final_df.to_csv(os.path.join(data_dir, "epl_2026.csv"), index=False)
    print("Saved epl_2026.csv")

if __name__ == "__main__":
    process_data()