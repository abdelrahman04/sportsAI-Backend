from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import uvicorn
import traceback

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

class WeightedKMeans:
    def __init__(self, n_clusters=5, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        
    def fit_predict(self, X, weights):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Add small epsilon to weights to prevent division by zero
        weights = np.maximum(weights, 1e-10)
        
        # Initialize centroids randomly
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices]
        
        for _ in range(self.max_iter):
            # Calculate distances to centroids
            distances = np.zeros((n_samples, self.n_clusters))
            for i in range(self.n_clusters):
                # Apply weights to the distance calculation
                weighted_diff = weights * (X - self.centroids[i])
                distances[:, i] = np.sqrt(np.sum(weighted_diff ** 2, axis=1))
            
            # Assign clusters
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    # Calculate weighted mean for each feature
                    for j in range(n_features):
                        # Use the same weight for all samples in the cluster
                        cluster_weights = np.ones(len(cluster_points)) * weights[j]
                        new_centroids[i, j] = np.average(cluster_points[:, j], weights=cluster_weights)
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
            
        return labels

class SOM:
    def __init__(self, grid_size=(5, 5), learning_rate=0.5, sigma=1.0, n_iterations=100, random_state=42):
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        
    def _initialize_weights(self, n_features):
        np.random.seed(self.random_state)
        self.weights = np.random.rand(self.grid_size[0], self.grid_size[1], n_features)
        
    def _get_bmu(self, x):
        # Find the Best Matching Unit (BMU)
        distances = np.sum((self.weights - x) ** 2, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx
    
    def _get_neighborhood(self, bmu_idx, iteration):
        # Calculate the neighborhood radius
        radius = self.sigma * np.exp(-iteration / self.n_iterations)
        
        # Create a grid of distances from BMU
        x, y = np.meshgrid(np.arange(self.grid_size[0]), np.arange(self.grid_size[1]))
        distances = np.sqrt((x - bmu_idx[0])**2 + (y - bmu_idx[1])**2)
        
        # Calculate neighborhood function
        neighborhood = np.exp(-distances**2 / (2 * radius**2))
        return neighborhood
    
    def fit_predict(self, X, weights):
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        # Apply weights to the data
        X_weighted = X * weights
        
        for iteration in range(self.n_iterations):
            # Update learning rate and sigma
            current_learning_rate = self.learning_rate * np.exp(-iteration / self.n_iterations)
            
            # Randomly select a sample
            idx = np.random.randint(n_samples)
            x = X_weighted[idx]
            
            # Find BMU
            bmu_idx = self._get_bmu(x)
            
            # Get neighborhood
            neighborhood = self._get_neighborhood(bmu_idx, iteration)
            
            # Update weights
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    influence = neighborhood[i, j] * current_learning_rate
                    self.weights[i, j] += influence * (x - self.weights[i, j])
        
        # Assign clusters
        labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            bmu_idx = self._get_bmu(X_weighted[i])
            labels[i] = bmu_idx[0] * self.grid_size[1] + bmu_idx[1]
            
        return labels

@app.post("/analyze-players")
async def analyze_players(request: PlayerAnalysisRequest):
    try:
        print("\n=== Starting Player Analysis ===")
        print(f"Request received - Position: {request.position}, Team: {request.team}")
        print(f"Specific role columns: {request.specific_role_cols}")

        # Validate position
        if request.position not in roles:
            print(f"Error: Invalid position {request.position}")
            raise HTTPException(status_code=400, detail=f"Invalid position. Must be one of: {list(roles.keys())}")
        
        # Validate team
        if request.team not in teams:
            print(f"Error: Invalid team {request.team}")
            raise HTTPException(status_code=400, detail=f"Invalid team. Must be one of: {teams}")
        
        # Validate specific_role_cols
        for col in request.specific_role_cols:
            if col not in attribute_map.values():
                print(f"Error: Invalid specific_role_col {col}")
                raise HTTPException(status_code=400, detail=f"Invalid specific_role_col: {col}")

        print("\n=== Loading Data ===")
        if request.position == "Goalkeeping":
            print("Loading goalkeeper data...")
            df = pd.read_csv('Players GK Merged.csv')
        else:
            print("Loading outfield player data...")
            df = pd.read_csv('Players Merged.csv')
        print(f"Data loaded successfully. Shape: {df.shape}")

        # Filter positions
        print("\n=== Filtering Positions ===")
        filtered_positions = df_position_roles[request.position]
        print(f"Filtering for positions: {filtered_positions}")
        df_filtered = df[df['Pos'].isin(filtered_positions)]
        print(f"After position filtering. Shape: {df_filtered.shape}")

        # Select numeric columns
        print("\n=== Processing Numeric Columns ===")
        numeric_cols = df_filtered.select_dtypes(include=['int64', 'float64']).columns
        print(f"Found {len(numeric_cols)} numeric columns")
        X = df_filtered[numeric_cols].fillna(0)
        print("Missing values filled with 0")

        # Scale the data
        print("\n=== Scaling Data ===")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("Data scaled successfully")

        # Get role stats
        print("\n=== Processing Role Stats ===")
        role_stats = roles[request.position]
        filtered_cols = [attribute_map[stat] for stat in role_stats]
        print(f"Role stats processed: {len(filtered_cols)} columns")

        # Create weights
        print("\n=== Creating Weights ===")
        weights = {col: 2.0 if col in request.specific_role_cols else 1.0 if col in filtered_cols else 0.0 for col in numeric_cols}
        weight_vector = np.ones(X.shape[1])
        for col, weight in weights.items():
            if col in numeric_cols:
                idx = list(numeric_cols).index(col)
                weight_vector[idx] = weight
        print("Weights created successfully")

        # Calculate distance matrix
        print("\n=== Calculating Distance Matrix ===")
        distance_matrix = custom_distance_matrix(X_scaled, weight_vector)
        print(f"Distance matrix shape: {distance_matrix.shape}")

        # Apply all four clustering methods
        print("\n=== Running Clustering Algorithms ===")
        dbscan = DBSCAN(eps=2.0, min_samples=5, metric='precomputed')
        gmm = GaussianMixture(n_components=5, random_state=42)
        wkmeans = WeightedKMeans(n_clusters=5, random_state=42)
        som = SOM(grid_size=(5, 5), learning_rate=0.5, sigma=1.0, n_iterations=100, random_state=42)
        
        print("Running DBSCAN...")
        dbscan_clusters = dbscan.fit_predict(distance_matrix)
        print(f"DBSCAN found {len(np.unique(dbscan_clusters))} clusters")
        
        print("Running GMM...")
        gmm_clusters = gmm.fit_predict(distance_matrix)
        print(f"GMM found {len(np.unique(gmm_clusters))} clusters")
        
        print("Running Weighted K-Means...")
        wkmeans_clusters = wkmeans.fit_predict(X_scaled, weight_vector)
        print(f"Weighted K-Means found {len(np.unique(wkmeans_clusters))} clusters")
        
        print("Running SOM...")
        som_clusters = som.fit_predict(X_scaled, weight_vector)
        print(f"SOM found {len(np.unique(som_clusters))} clusters")

        # Get team data
        print("\n=== Processing Team Data ===")
        team_data = df_filtered[df_filtered['Squad'] == request.team]
        if len(team_data) == 0:
            print(f"Error: No data found for team {request.team}")
            raise HTTPException(status_code=404, detail=f"No data found for team: {request.team}")
        print(f"Found {len(team_data)} players for team {request.team}")

        # Calculate average profile
        print("\n=== Calculating Average Profile ===")
        relevant_cols = []
        for attr in roles[request.position]:
            if attr in attribute_map:
                # Special handling for Cmp% based on position
                if attr == "Completed Passes Total" and request.position != "Goalkeeping":
                    relevant_cols.append("Cmp%")
                elif attr == "Passes Completed (Launched)" and request.position == "Goalkeeping":
                    relevant_cols.append("Cmp%")
                else:
                    relevant_cols.append(attribute_map[attr])
        
        relevant_cols = [col for col in relevant_cols if col in numeric_cols]
        relevant_indices = [list(numeric_cols).index(col) for col in relevant_cols]
        relevant_weights = weight_vector[relevant_indices]
        average_profile = team_data[team_data["MP"] > 20][relevant_cols].mean()

        # Convert abbreviated attribute names to full forms
        reverse_attribute_map = {v: k for k, v in attribute_map.items()}
        full_form_profile = {}
        for abbrev, value in average_profile.items():
            if abbrev in reverse_attribute_map:
                if abbrev == "Cmp%":
                    if request.position == "Goalkeeping":
                        full_form_profile["Passes Completed (Launched)"] = float(value)
                    else:
                        full_form_profile["Completed Passes Total"] = float(value)
                else:
                    full_form_profile[reverse_attribute_map[abbrev]] = float(value)
            else:
                full_form_profile[abbrev] = float(value)

        print("Average profile calculated with full attribute names")

        # Find similar players using all methods
        def get_top_players(clusters, method_name):
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
                    "age": int(row['Age']),
                    "distance": float(distance),
                    "stats": {stat: float(row[attribute_map[stat]]) for stat in roles[request.position] if stat in attribute_map and attribute_map[stat] in row},
                    "method": method_name
                })
            return sorted(player_distances, key=lambda x: x['distance'])[:5]

        # Get top players from all methods
        dbscan_players = get_top_players(dbscan_clusters, "DBSCAN")
        gmm_players = get_top_players(gmm_clusters, "GMM")
        wkmeans_players = get_top_players(wkmeans_clusters, "WeightedKMeans")
        som_players = get_top_players(som_clusters, "SOM")

        # Find common players between all methods
        dbscan_player_names = {p['player'] for p in dbscan_players}
        gmm_player_names = {p['player'] for p in gmm_players}
        wkmeans_player_names = {p['player'] for p in wkmeans_players}
        som_player_names = {p['player'] for p in som_players}
        
        # Get players that appear in at least two methods
        common_players = set()
        all_methods = [dbscan_player_names, gmm_player_names, wkmeans_player_names, som_player_names]
        for i in range(len(all_methods)):
            for j in range(i + 1, len(all_methods)):
                common_players.update(all_methods[i].intersection(all_methods[j]))

        print(f"Found {len(common_players)} common players across methods")
        print("Common players:", common_players)

        # Get full player info for common players
        print("\n=== Preparing Response ===")
        common_player_info = []
        for player in dbscan_players:
            if player['player'] in common_players:
                common_player_info.append(player)

        print(f"Final response includes {len(common_player_info)} players")
        print(dbscan_player_names)
        print(gmm_player_names)
        print(wkmeans_player_names)
        print(som_player_names)
        print("\n=== Analysis Complete ===")
        return {
            "position": request.position,
            "team": request.team,
            "specific_role_cols": request.specific_role_cols,
            "team_players": team_data[team_data["MP"] > 20]['Player'].tolist(),
            "similar_players": common_player_info,
            "dbscan_players": dbscan_players,
            "gmm_players": gmm_players,
            "wkmeans_players": wkmeans_players,
            "som_players": som_players,
            "average_profile": full_form_profile
        }

    except Exception as e:
        print("\n=== ERROR OCCURRED ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Stack trace:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 