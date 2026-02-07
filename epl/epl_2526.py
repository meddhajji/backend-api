import soccerdata as sd
import pandas as pd
import numpy as np
import warnings


fbref = sd.FBref(leagues = "ENG-Premier League", seasons = "25-26")
xg = fbref.read_team_match_stats(stat_type = "schedule")
xg = xg.reset_index() 
xg.columns = xg.columns.str.lower().str.replace(' ', '_')
xg = xg.drop(columns=['league', 'game', 'season', 'time', 'day', 'xg','xga','poss',
'attendance','captain','formation','opp_formation','referee','match_report', 'notes'], errors='ignore')


import os

# ... imports ...

# ... code ...

output_path = os.path.join(os.path.dirname(__file__), 'data', 'epl_2526.csv')
xg.to_csv(output_path, index=False)
