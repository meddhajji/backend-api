import soccerdata as sd
import pandas as pd
import numpy as np
import warnings

fb = sd.FBref(leagues = "ENG-Premier League", seasons = "20-21")
xg = fb.read_team_match_stats(stat_type = "schedule")
xg = xg.reset_index() 

fb = sd.FBref(leagues = "ENG-Premier League", seasons = "21-22")
xg1 = fb.read_team_match_stats(stat_type = "schedule")
xg1 = xg1.reset_index() 

fb = sd.FBref(leagues = "ENG-Premier League", seasons = "22-23")
xg2 = fb.read_team_match_stats(stat_type = "schedule")
xg2 = xg2.reset_index() 

fb = sd.FBref(leagues = "ENG-Premier League", seasons = "23-24")
xg3 = fb.read_team_match_stats(stat_type = "schedule")
xg3 = xg3.reset_index() 

fb = sd.FBref(leagues = "ENG-Premier League", seasons = "24-25")
xg4 = fb.read_team_match_stats(stat_type = "schedule")
xg4 = xg4.reset_index() 


epl_2025 = pd.concat([xg, xg1, xg2, xg3, xg4], ignore_index=True)
epl_2025.columns = epl_2025.columns.str.lower().str.replace(' ', '_')
epl_2025 = epl_2025.drop(columns=['league', 'game', 'season', 'time', 'day', 'xg','xga','poss',
'attendance','captain','formation','opp_formation','referee','match_report', 'notes'], errors='ignore')

epl_2025.to_csv(r"epl_2025.csv", index=False)
