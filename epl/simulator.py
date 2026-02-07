import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from collections import deque
from tqdm import tqdm # Added for progress bar
from fuzzywuzzy import process # Added for interactive match

# --- CONFIGURATION ---
K_FACTOR = 30
HOME_ADV = 100
SIMULATIONS = 300
MODEL_FEATURES = ['home_elo_pre', 'away_elo_pre', 'elo_diff', 
                  'home_roll_pts_5', 'away_roll_pts_5', 
                  'home_venue_roll_pts_3', 'away_venue_roll_pts_3']
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'epl_model_new.pkl')
CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'epl_2026.csv')

class SeasonSimulator:
    def __init__(self, df, init_state=None):
        """
        Lightweight Simulator.
        init_state: Optional dict containing pre-computed 'standings', 'elo', 'history'.
                    If None, it calculates them from the provided DataFrame (played matches).
        """
        self.teams = sorted(pd.concat([df['team'], df['opponent']]).unique())
        self.team_map = {t: i for i, t in enumerate(self.teams)}
        self.n_teams = len(self.teams)
        
        if init_state:
            # Deep copy mutable structures for independent simulation
            self.standings = init_state['standings'].copy()
            self.elo = init_state['elo'].copy()
            self.history_global = [deque(h, maxlen=5) for h in init_state['history_global']]
            self.history_home = [deque(h, maxlen=3) for h in init_state['history_home']]
            self.history_away = [deque(h, maxlen=3) for h in init_state['history_away']]
        else:
            self._init_from_df(df)

    def _init_from_df(self, df):
        """Build initial state by parsing played matches just once."""
        self.standings = np.zeros((self.n_teams, 3), dtype=int) # [Pts, GF, GA]
        self.elo = np.full(self.n_teams, 1300.0) # Default, will be overwritten
        self.history_global = [deque(maxlen=5) for _ in range(self.n_teams)]
        self.history_home = [deque(maxlen=3) for _ in range(self.n_teams)]
        self.history_away = [deque(maxlen=3) for _ in range(self.n_teams)]

        # 1. Elo & Form from History
        # We assume df contains verified results for the current season
        played = df[df['result'].notna()].sort_values('date')
        
        # Populate Standings & History
        for _, row in played.iterrows():
            if row['team'] not in self.team_map or row['opponent'] not in self.team_map: continue
            
            h_idx, a_idx = self.team_map[row['team']], self.team_map[row['opponent']]
            result = row['result']
            hg, ag = int(row['gf']), int(row['ga'])
            
            # Points
            if result == 'W': h_pts, a_pts = 3, 0
            elif result == 'D': h_pts, a_pts = 1, 1
            else: h_pts, a_pts = 0, 3
            
            self.standings[h_idx] += [h_pts, hg, ag]
            self.standings[a_idx] += [a_pts, ag, hg]
            
            self.history_global[h_idx].append(h_pts)
            self.history_global[a_idx].append(a_pts)
            self.history_home[h_idx].append(h_pts)
            self.history_away[a_idx].append(a_pts)
            
            self._update_elo_internal(h_idx, a_idx, result)

    def _update_elo_internal(self, h_idx, a_idx, result):
        res_num = {'W': 1, 'D': 0.5, 'L': 0}[result]
        h_elo, a_elo = self.elo[h_idx], self.elo[a_idx]
        expected = 1 / (10**(-(h_elo + HOME_ADV - a_elo) / 400) + 1)
        change = K_FACTOR * (res_num - expected)
        self.elo[h_idx] += change
        self.elo[a_idx] -= change

    def get_features_batch(self, homes, aways):
        rows = []
        for h, a in zip(homes, aways):
            # Rolling Calcs
            hg, ag = self.history_global[h], self.history_global[a]
            hh, aa = self.history_home[h], self.history_away[a]
            
            h_roll = sum(hg)/len(hg) if hg else 1.3
            a_roll = sum(ag)/len(ag) if ag else 1.3
            h_venue = sum(hh)/len(hh) if hh else 1.5
            a_venue = sum(aa)/len(aa) if aa else 1.0
            
            rows.append([
                self.elo[h], self.elo[a], self.elo[h] - self.elo[a],
                h_roll, a_roll, h_venue, a_venue
            ])
        return np.array(rows)

    def update_match(self, h_idx, a_idx, result, hg, ag):
        # 1. Standings
        if result == 'W': h_pts, a_pts = 3, 0
        elif result == 'D': h_pts, a_pts = 1, 1
        else: h_pts, a_pts = 0, 3
        
        self.standings[h_idx] += [h_pts, hg, ag]
        self.standings[a_idx] += [a_pts, ag, hg]
        
        # 2. History
        self.history_global[h_idx].append(h_pts)
        self.history_global[a_idx].append(a_pts)
        self.history_home[h_idx].append(h_pts)
        self.history_away[a_idx].append(a_pts)
        
        # 3. Elo
        self._update_elo_internal(h_idx, a_idx, result)

    def get_snapshot(self):
        return {
            'elo': self.elo.copy(),
            'standings': self.standings.copy(),
            'history_global': [list(x) for x in self.history_global],
            'history_home': [list(x) for x in self.history_home],
            'history_away': [list(x) for x in self.history_away]
        }

# --- GLOBAL LOADERS ---
def train_model(df):
    print("Training new model...")
    # Map W/D/L
    train_df = df[df['result'].notna()].copy()
    if train_df.empty: return None
    
    target_map = {'W': 0, 'D': 1, 'L': 2}
    train_df = train_df[train_df['result'].isin(target_map.keys())]
    train_df['target'] = train_df['result'].map(target_map)
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    # Check if features exist
    for f in MODEL_FEATURES:
        if f not in train_df.columns:
            print(f"Missing feature: {f}")
            return None
            
    model.fit(train_df[MODEL_FEATURES], train_df['target'])
    try:
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    except:
        pass
    return model

def load_assets():
    if not os.path.exists(CSV_PATH): 
        print(f"CSV not found at {CSV_PATH}")
        return None, None
        
    df = pd.read_csv(CSV_PATH)
    # Ensure season column
    if 'season' not in df.columns:
        df['season'] = df['match_id'].apply(lambda x: int(x.split('_')[0]) if isinstance(x, str) else 2026)
        
    try:
        model = joblib.load(MODEL_PATH)
    except:
        model = train_model(df)
        
    return df, model

# --- API METHODS ---

def run_simulation_api(df, model, features, forced_results=None, simulations=100):
    if forced_results is None: forced_results = {}
    
    # 1. Setup Base State
    current_df = df[df['season'] == 2026].copy()
    base_sim = SeasonSimulator(current_df)
    base_state = base_sim.get_snapshot()
    
    # 2. Prepare Schedule (Unplayed Games)
    # Group by 'round' or chunks for batching
    unplayed = current_df[current_df['result'].isna()].sort_values('date')
    
    # Use 'round' column if available, otherwise fake chunks
    if 'round' not in unplayed.columns:
        unplayed['round'] = np.arange(len(unplayed)) // 10 # Approx 10 games per round

    schedule_chunks = []
    for _, chunk in unplayed.groupby('round'):
        # Map match chunk to indices
        matches = []
        for _, row in chunk.iterrows():
            if row['team'] in base_sim.team_map and row['opponent'] in base_sim.team_map:
                matches.append((
                    base_sim.team_map[row['team']], 
                    base_sim.team_map[row['opponent']]
                ))
        if matches:
            schedule_chunks.append(matches)
            
    # Forced lookup
    forced_map = {}
    for (h_name, a_name), (h_g, a_g) in forced_results.items():
        if h_name in base_sim.team_map and a_name in base_sim.team_map:
            forced_map[(base_sim.team_map[h_name], base_sim.team_map[a_name])] = (h_g, a_g)
            
    # Capture Base Elo
    current_elos = {}
    for i, team in enumerate(base_sim.teams):
        current_elos[team] = int(base_sim.elo[i])

    # 3. Monte Carlo Loop
    results_accum = []
    
    iterator = range(simulations)
    if __name__ == "__main__":
        iterator = tqdm(range(simulations), desc="Simulating")

    for _ in iterator:
        sim = SeasonSimulator(current_df, init_state=base_state)
        
        for chunk in schedule_chunks:
            # Separate forced vs unforced in this chunk
            unforced_homes = []
            unforced_aways = []
            
            # 1. Process forced matches immediately
            # 2. Collect unforced for batch predict
            for h_idx, a_idx in chunk:
                if (h_idx, a_idx) in forced_map:
                    hg, ag = forced_map[(h_idx, a_idx)]
                    res = 'W' if hg > ag else ('L' if hg < ag else 'D')
                    sim.update_match(h_idx, a_idx, res, hg, ag)
                else:
                    unforced_homes.append(h_idx)
                    unforced_aways.append(a_idx)
            
            if not unforced_homes:
                continue
                
            # BATCH PREDICTION
            feats = sim.get_features_batch(unforced_homes, unforced_aways)
            probs = model.predict_proba(feats) # Matrix [N, 3]
            
            # Vectorized Sim
            n_batch = len(probs)
            rolls = np.random.rand(n_batch)
            
            # Vectorized outcomes
            outcomes = np.full(n_batch, 'D', dtype=object)
            outcomes[rolls < probs[:, 0]] = 'W'
            outcomes[rolls >= (probs[:, 0] + probs[:, 1])] = 'L'
            
            # Vectorized Scores (Poisson)
            lam_h = 0.5 + 3 * probs[:, 0]
            lam_a = 0.5 + 3 * probs[:, 2]
            h_goals = np.random.poisson(lam_h)
            a_goals = np.random.poisson(lam_a)
            
            # Adjust scores loop
            for k in range(n_batch):
                res = outcomes[k]
                hg, ag = h_goals[k], a_goals[k]
                
                if res == 'W' and hg <= ag: hg = ag + 1
                elif res == 'L' and ag <= hg: ag = hg + 1
                elif res == 'D': hg = ag
                
                sim.update_match(unforced_homes[k], unforced_aways[k], res, hg, ag)
            
        # End of Season
        table = []
        for i, team in enumerate(sim.teams):
            p, gf, ga = sim.standings[i]
            table.append({'Team': team, 'Pts': p, 'GD': gf-ga, 'GF': gf})
        results_accum.append(table)

    return _aggregate_results(results_accum, current_elos)

def _aggregate_results(results_collection, elo_map=None):
    team_stats = {}
    for season_table in results_collection:
        season_table.sort(key=lambda x: (x['Pts'], x['GD'], x['GF']), reverse=True)
        for rank_idx, row in enumerate(season_table):
            t = row['Team']
            if t not in team_stats:
                team_stats[t] = {'pts': [], 'gd': [], 'gf': [], 'ranks': []}
            team_stats[t]['pts'].append(row['Pts'])
            team_stats[t]['gd'].append(row['GD'])
            team_stats[t]['gf'].append(row['GF'])
            team_stats[t]['ranks'].append(rank_idx + 1)
            
    final_output = []
    for team, stats in team_stats.items():
        n = len(stats['pts'])
        ranks = np.array(stats['ranks'])
        final_output.append({
            'Team': team,
            'Avg_Pts': round(np.mean(stats['pts']), 1),
            'Avg_GD': round(np.mean(stats['gd']), 1),
            'Avg_GF': round(np.mean(stats['gf']), 1),
            'Avg_Rank': round(np.mean(ranks), 1),
            'Title %': round(np.sum(ranks == 1)/n * 100, 1),
            'Top 4 %': round(np.sum(ranks <= 4)/n * 100, 1),
            'Relegation %': round(np.sum(ranks >= 18)/n * 100, 1),
            'Min_Pts': int(np.min(stats['pts'])),
            'Max_Pts': int(np.max(stats['pts'])),
            'Current_Elo': int(elo_map.get(team, 0)) if elo_map else 0,
            'Form': [] 
        })
        
    final_output.sort(key=lambda x: (x['Avg_Pts'], x['Avg_GD']), reverse=True)
    return final_output

def get_dashboard_initial_data(df, model, features):
    # Just read raw DF
    current = df[df['season'] == 2026].copy()
    played = current[current['result'].notna()].sort_values('date', ascending=False)
    unplayed = current[current['result'].isna()].sort_values('date', ascending=True)
    
    # Last Gameweek (Unique teams, approx 10 matches)
    last_gw = []
    seen = set()
    for _, row in played.iterrows():
        if row['team'] in seen or row['opponent'] in seen: continue
        seen.add(row['team']); seen.add(row['opponent'])
        last_gw.append({
            'Home': row['team'], 'Away': row['opponent'], 
            'Score': f"{int(row['gf'])}-{int(row['ga'])}"
        })
        if len(last_gw) >= 10: break
        
    # Next Gameweek (Predict)
    # Need Simulator instance to get Current Elo/Features for prediction
    sim = SeasonSimulator(current)
    
    next_gw = []
    seen_next = set()
    for _, row in unplayed.iterrows():
        if row['team'] in seen_next or row['opponent'] in seen_next: continue
        
        if row['team'] not in sim.team_map or row['opponent'] not in sim.team_map: continue
        
        h_idx, a_idx = sim.team_map[row['team']], sim.team_map[row['opponent']]
        
        seen_next.add(row['team']); seen_next.add(row['opponent'])
        
        feats = sim.get_features_batch([h_idx], [a_idx])
        probs = model.predict_proba(feats)[0]
        
        next_gw.append({
            'Home': row['team'], 'Away': row['opponent'],
            'Home Win %': float(probs[0]*100),
            'Draw %': float(probs[1]*100),
            'Away Win %': float(probs[2]*100)
        })
        if len(next_gw) >= 10: break
        
    return {
        'last_gw': last_gw,
        'next_gw': next_gw,
        'teams': sim.teams,
        'unplayed_matches': unplayed[['team', 'opponent']].to_dict(orient='records')
    }

def get_insights(results_list, df=None, features=None):
    # Simple Insights based on results
    df_results = pd.DataFrame(results_list)
    return {
        'overachievers': df_results.head(5).to_dict(orient='records'),
        'underachievers': df_results.tail(5).to_dict(orient='records'),
        'title_contenders': df_results[df_results['Title %'] > 0].to_dict(orient='records')
    }

# --- CLI INTERACTION ---
def get_user_forced_results(teams, df):
    forced = {}
    print("\n" + "="*80)
    print("FORCED RESULTS INTERFACE")
    print("="*80)
    
    while True:
        choice = input("\nAdd forced result? (y/n): ").lower().strip()
        if choice != 'y': break
        
        h_in = input("Home Team: ").strip()
        h_match, h_score = process.extractOne(h_in, teams)
        if h_score < 70:
            print(f"Team '{h_in}' not found!")
            continue
        print(f"Selected: {h_match}")
        
        a_in = input("Away Team: ").strip()
        a_match, a_score = process.extractOne(a_in, teams)
        if a_score < 70:
            print(f"Team '{a_in}' not found!")
            continue
        print(f"Selected: {a_match}")
        
        if h_match == a_match:
            print("Home and away teams must be different!")
            continue

        # Validation: Check if match played
        # We need to find the specific match row in the DF (Season 2026)
        match_record = df[
            (df['season'] == 2026) & 
            (df['team'] == h_match) & 
            (df['opponent'] == a_match)
        ]
        
        if match_record.empty:
            print(f"Match {h_match} vs {a_match} not found in this season's fixtures.")
            continue
            
        row = match_record.iloc[0]
        if pd.notna(row['result']):
            score_str = f"{int(row['gf'])}-{int(row['ga'])}"
            print(f"This match was already played on {row['date'].split()[0]}")
            print(f"Result: {h_match} {score_str} {a_match}")
            continue

        score = input("Score (e.g., 2-1): ").strip()
        try:
            hg, ag = map(int, score.split('-'))
            forced[(h_match, a_match)] = (hg, ag)
            print(f"Added: {h_match} {hg}-{ag} {a_match}")
        except:
            print("Invalid score format.")
            
    return forced

def main():
    print("Loading assets...")
    df, model = load_assets()
    if df is None: return
    
    print(f"Data Loaded: {len(df)} matches")
    
    # Dashboard Preview
    print("\n" + "="*80)
    print("LAST GAMEWEEK")
    print("="*80)
    dash = get_dashboard_initial_data(df, model, MODEL_FEATURES)
    for m in dash['last_gw']:
        print(f"{m['Home']} {m['Score']} {m['Away']}")
        
    print("\n" + "="*80)
    print("NEXT FIXTURES PREDICTION")
    print("="*80)
    for m in dash['next_gw']:
        print(f"{m['Home']} vs {m['Away']} (H: {m['Home Win %']:.1f}% D: {m['Draw %']:.1f}% A: {m['Away Win %']:.1f}%)")
        
    # Interactive Mode
    forced = get_user_forced_results(dash['teams'], df)
    
    print(f"\nRunning {SIMULATIONS} simulations...")
    results = run_simulation_api(df, model, MODEL_FEATURES, forced_results=forced, simulations=SIMULATIONS)
    
    print("\n" + "="*90)
    print("FINAL PREDICTED TABLE")
    print("="*90)
    res_df = pd.DataFrame(results)
    # Show all relevant columns
    cols = ['Team', 'Avg_Pts', 'Avg_GD', 'Avg_GF', 'Title %', 'Top 4 %', 'Relegation %', 'Avg_Rank', 'Min_Pts', 'Max_Pts']
    print(res_df[cols].to_string(index=False))

if __name__ == "__main__":
    main()