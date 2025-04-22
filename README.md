# SportsAI - Football Player Analysis API

A powerful FastAPI-based backend service for comprehensive football player analysis, providing insights based on positions, teams, and specific role attributes.

## Features

- **Position-Based Analysis**: Analyze players based on their specific positions
- **Team Context**: Consider team context in player evaluations
- **Customizable Attributes**: Define and analyze role-specific attributes
- **Similar Player Recommendations**: Find players with similar characteristics
- **Comprehensive Position Coverage**:
  - Forward
  - Winger
  - Attacking Midfielder
  - Central Midfielder
  - Fullback
  - Central Defender
  - Goalkeeper

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sportsAI-Backend.git
cd sportsAI-Backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python api.py
```

The server will start on http://localhost:8000

## API Documentation

### Key Endpoints

#### 1. POST /analyze-players
Analyzes players based on position, team, and specific role attributes.

**Request Body:**
```json
{
    "position": "string",
    "team": "string",
    "specific_role_cols": ["string"]
}
```

#### 2. POST /analyze-player-to-player
Compares a specific player with others to find similar players based on their attributes.

**Request Body:**
```json
{
    "player_name": "string",
    "team": "string",
    "specific_role_cols": ["string"],
    "specific_role_weight": 5.0
}
```

## Analysis Methods

### Distance Calculation Methods

1. **Weighted Euclidean Distance**
   - Primary distance metric for player comparison
   - Weights are position-specific and attribute-specific
   - Normalized to account for different scales of attributes

2. **Custom Distance Matrix**
   - Computes pairwise distances between all players
   - Used for clustering and similarity analysis
   - Optimized for large datasets

### Clustering Models

1. **DBSCAN (Density-Based Spatial Clustering)**
   - Identifies clusters based on density
   - Effective for finding natural player groupings

2. **Gaussian Mixture Model (GMM)**
   - Probabilistic clustering approach
   - Handles overlapping player characteristics

3. **Weighted K-Means**
   - Custom implementation with position-specific weights
   - Considers attribute importance in clustering

4. **Self-Organizing Map (SOM)**
   - Neural network-based clustering
   - 5x5 grid for player organization

All models yielded same/similar results

### Distance Calculation and Weighting

1. **Base Distance Calculation**
   - Uses weighted Euclidean distance
   - Normalized using StandardScaler

2. **Attribute Weighting System**
   - Default weights: 1.0 for position-specific, 0.5 for others
   - Custom weights: k multiplier for user-selected attributes
   - Position-specific attribute importance

3. **Team Context Integration**
   - Calculates team average profile (>20 matches played)
   - Applies team-specific weights

4. **Final Player Selection**
   - Combines individual and team distances
   - Ranks players by weighted similarity
   - Returns top 5 most similar players
   - Includes detailed performance metrics


## Data Sources

The analysis utilizes two primary datasets:

1. **Players Merged.csv**
   - Comprehensive data for outfield players
   - Includes performance metrics and statistics
   - Position-specific attributes

2. **Players GK Merged.csv**
   - Specialized data for goalkeepers
   - Goalkeeper-specific metrics and statistics

## Data Processing Pipeline

### 1. Data Collection
- FBref data extraction via Excel exports
- Transfermarkt web scraping for precise player positions
- Integration of multiple data sources

### 2. Data Cleaning
- Data type standardization
- Irregular row processing
- League and team name standardization
- Web Scraping to get specific player positions
- Position data validation and correction

### 3. Dataset Integration
- Creation of unified player datasets
- Team dataset integration
- Data type optimization
- Cross-reference validation
