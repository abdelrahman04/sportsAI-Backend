from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import uvicorn

app = FastAPI(
    title="Football Player Analysis API",
    description="API for analyzing football players based on their positions, teams, and specific role attributes",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows requests from your frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the request model
class PlayerAnalysisRequest(BaseModel):
    position: str
    team: str
    specific_role_cols: List[str]

# Load data and define mappings (moved from main.py)
teams = [
    # Premier League
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Burnley",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Liverpool", "Luton Town",
    "Manchester City", "Manchester Utd", "Newcastle Utd", "Nott'ham Forest",
    "Sheffield Utd", "Tottenham", "West Ham", "Wolves",
    
    # La Liga
    "Alavés", "Almería", "Athletic Club", "Atlético Madrid", "Barcelona", "Betis",
    "Cádiz", "Celta Vigo", "Getafe", "Girona", "Granada", "Las Palmas",
    "Mallorca", "Osasuna", "Rayo Vallecano", "Real Madrid", "Real Sociedad",
    "Sevilla", "Valencia", "Villarreal",
    
    # Bundesliga
    "Augsburg", "Bayern Munich", "Bochum", "Darmstadt 98", "Dortmund",
    "Eint Frankfurt", "Freiburg", "Gladbach", "Heidenheim", "Hoffenheim",
    "Köln", "Leverkusen", "Mainz 05", "RB Leipzig", "Stuttgart", "Union Berlin",
    "Werder Bremen", "Wolfsburg",
    
    # Serie A
    "Atalanta", "Bologna", "Cagliari", "Empoli", "Fiorentina", "Frosinone",
    "Genoa", "Hellas Verona", "Inter", "Juventus", "Lazio", "Lecce", "Milan",
    "Monza", "Napoli", "Roma", "Salernitana", "Sassuolo", "Torino", "Udinese",
    
    # Ligue 1
    "Brest", "Clermont Foot", "Le Havre", "Lens", "Lille", "Lorient", "Lyon",
    "Marseille", "Metz", "Monaco", "Montpellier", "Nantes", "Nice", "Paris S-G",
    "Reims", "Rennes", "Strasbourg", "Toulouse"
]

attribute_map = {
    "Goals": "Gls",
    "Shots on Target": "SoT", 
    "Shots On Target Per 90": "SoT/90",
    "Goals/Shots on target": "G/SoT",
    "Penalty Kicks Made": "PK",
    "xG (Expected Goals)": "xG",
    "Shot-creating actions": "SCA",
    "Shots from Freekick": "FK",
    "Assists": "Ast",
    "xA (Expected Assists)": "xA",
    "Key Passes": "KP",
    "Crosses into Penalty Area": "CrsPA",
    "Progressive Passes": "PrgP",
    "Completed Passes Total": "Cmp%",
    "Passes into Penalty Area": "PPA",
    "Successful Take-On%": "Succ%",
    "Carries": "Carries",
    "Tackles Won": "TklW",
    "% of Dribblers Tackled": "Tkl%",
    "Blocks": "Blocks",
    "Passes Block": "Pass",
    "Interceptions": "Int",
    "Clearances": "Clr", 
    "Ball Recoveries": "Recov",
    "% of Aerial Duels Won": "Won%",
    "Goals Against /90": "GA90",
    "Save Percentage": "Save%",
    "Clean Sheet Percentage": "CS%",
    "Penalty Kicks Saved %": "Save%2",
    "Passes Completed (Launched)": "Cmp%",
    "Crosses Stopped": "Stp%",
    "% of Passes that were Launched": "Launch%"
}

roles = {
    "Forward": ["Goals", "Shots on Target", "Shots On Target Per 90", "Goals/Shots on target", "Penalty Kicks Made", "xG (Expected Goals)", "Shot-creating actions"],
    "Winger": ["Goals", "Shots on Target", "Shots On Target Per 90", "Goals/Shots on target", "Penalty Kicks Made", "xG (Expected Goals)", "Shots from Freekick", "Assists", "xA (Expected Assists)", "Key Passes", "Crosses into Penalty Area"],
    "Attacking Mid": ["Goals", "Shots on Target", "Shots On Target Per 90", "Goals/Shots on target", "Penalty Kicks Made", "xG (Expected Goals)", "Shots from Freekick", "Assists", "xA (Expected Assists)", "Key Passes", "Crosses into Penalty Area", "Progressive Passes", "Completed Passes Total", "Passes into Penalty Area"],
    "Centre Mid": ["Crosses into Penalty Area", "Progressive Passes", "Completed Passes Total", "Assists", "xA (Expected Assists)", "Successful Take-On%", "Carries", "Tackles Won", "% of Dribblers Tackled", "Blocks", "Passes Block", "Interceptions", "Clearances"],
    "Fullback": ["Tackles Won", "% of Dribblers Tackled", "Blocks", "Passes Block", "Interceptions", "Clearances", "Successful Take-On%", "Crosses into Penalty Area", "Progressive Passes", "Completed Passes Total", "Assists", "xA (Expected Assists)", "Carries", "Passes into Penalty Area", "Key Passes"],
    "Centre Defense": ["Tackles Won", "% of Dribblers Tackled", "Blocks", "Passes Block", "Interceptions", "Clearances", "Carries", "Successful Take-On%", "Ball Recoveries", "% of Aerial Duels Won"],
    "Goalkeeping": ["Goals Against /90", "Save Percentage", "Clean Sheet Percentage", "Penalty Kicks Saved %", "Passes Completed (Launched)", "Crosses Stopped", "% of Passes that were Launched", "Assists", "xA (Expected Assists)"]
}

df_position_roles = {
    "Forward": ["CF", "SS", "FW"],
    "Winger": ["LW", "RW"],
    "Attacking Mid": ["AM"],
    "Centre Mid": ["CM", "DM", "LM", "RM"],
    "Centre Defense": ["CB", "DF"],
    "Fullback": ["LB", "RB"],
    "Goalkeeping": ["GK"]
}

def weighted_euclidean_distance(X, Y, weights):
    return np.sqrt(np.sum(weights * (X - Y) ** 2))

def custom_distance_matrix(X, weights):
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            distance = weighted_euclidean_distance(X[i], X[j], weights)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix

@app.post("/analyze-players")
async def analyze_players(request: PlayerAnalysisRequest):
    try:
        # Validate position
        if request.position not in roles:
            raise HTTPException(status_code=400, detail=f"Invalid position. Must be one of: {list(roles.keys())}")
        
        # Validate team
        if request.team not in teams:
            raise HTTPException(status_code=400, detail=f"Invalid team. Must be one of: {teams}")
        
        # Validate specific_role_cols
        for col in request.specific_role_cols:
            if col not in attribute_map.values():
                raise HTTPException(status_code=400, detail=f"Invalid specific_role_col: {col}")
        if request.position == "Goalkeeping":
            df = pd.read_csv('Players GK Merged.csv')
        else:
            df = pd.read_csv('Players Merged.csv')

        # Filter positions
        filtered_positions = df_position_roles[request.position]
        df_filtered = df[df['Pos'].isin(filtered_positions)]

        # Select numeric columns (integers and floats)
        numeric_cols = df_filtered.select_dtypes(include=['int64', 'float64']).columns
        X = df_filtered[numeric_cols].fillna(0)

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Get role stats
        role_stats = roles[request.position]
        filtered_cols = [attribute_map[stat] for stat in role_stats]

        # Create weights
        weights = {col: 2.0 if col in request.specific_role_cols else 1.0 if col in filtered_cols else 0.0 for col in numeric_cols}
        weight_vector = np.ones(X.shape[1])
        for col, weight in weights.items():
            if col in numeric_cols:
                idx = list(numeric_cols).index(col)
                weight_vector[idx] = weight

        # Calculate distance matrix
        distance_matrix = custom_distance_matrix(X_scaled, weight_vector)

        # Apply DBSCAN
        dbscan = DBSCAN(eps=2.0, min_samples=5, metric='precomputed')
        clusters = dbscan.fit_predict(distance_matrix)
        df_filtered['Cluster'] = clusters

        # Get team data
        team_data = df_filtered[df_filtered['Squad'] == request.team]
        if len(team_data) == 0:
            raise HTTPException(status_code=404, detail=f"No data found for team: {request.team}")

        # Calculate average profile
        relevant_cols = [attribute_map[attr] for attr in roles[request.position]]
        relevant_cols = [col for col in relevant_cols if col in numeric_cols]
        relevant_indices = [list(numeric_cols).index(col) for col in relevant_cols]
        relevant_weights = weight_vector[relevant_indices]
        average_profile = team_data[team_data["MP"] > 20][relevant_cols].mean()
        # Find similar players
        player_distances = []
        for idx, row in df_filtered.iterrows():
            if row['Squad'] == request.team:
                continue
            player_profile = row[relevant_cols]
            distance = weighted_euclidean_distance(player_profile.values, average_profile.values, relevant_weights)
            player_distances.append({
                "player": row['Player'],
                "position": row['Pos'],
                "team": row['Squad'],
                "distance": float(distance),
                "stats": {stat: float(row[attribute_map[stat]]) for stat in roles[request.position] if stat in attribute_map and attribute_map[stat] in row}
            })

        # Sort and get top 5 matches
        top_matches = sorted(player_distances, key=lambda x: x['distance'])[:5]

        return {
            "position": request.position,
            "team": request.team,
            "specific_role_cols": request.specific_role_cols,
            "team_players": team_data[team_data["MP"] > 20]['Player'].tolist(),
            "similar_players": top_matches
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 