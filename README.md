# Football Player Analysis API

This API provides functionality to analyze football players based on their positions, teams, and specific role attributes. It uses machine learning to find similar players and provide detailed statistical analysis.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the required data files in the root directory:
- `Players Merged.csv`
- `Squad Merged.csv`

3. Run the API:
```bash
python api.py
```

The API will be available at `http://localhost:8000`

## API Documentation

### Endpoints

#### POST /analyze-players

Analyzes players based on position, team, and specific role attributes.

**Request Body:**
```json
{
    "position": "string",
    "team": "string",
    "specific_role_cols": ["string"]
}
```

**Parameters:**
- `position`: One of the following positions:
  - Forward
  - Winger
  - Attacking Mid
  - Centre Mid
  - Fullback
  - Centre Defense
  - Goalkeeping

- `team`: One of the following teams:
  - Arsenal
  - Aston Villa
  - Bournemouth
  - Brentford
  - Brighton
  - Burnley
  - Chelsea
  - Crystal Palace
  - Everton
  - Fulham
  - Liverpool
  - Luton Town
  - Manchester City
  - Manchester United
  - Newcastle United
  - Nottingham Forest
  - Sheffield United
  - Tottenham Hotspur
  - West Ham United
  - Wolverhampton Wanderers

- `specific_role_cols`: List of specific attributes to focus on. Must be valid attribute codes from the following list:
  - Gls (Goals)
  - SoT (Shots on Target)
  - SoT/90 (Shots On Target Per 90)
  - G/SoT (Goals/Shots on target)
  - PK (Penalty Kicks Made)
  - xG (Expected Goals)
  - SCA (Shot-creating actions)
  - FK (Shots from Freekick)
  - Ast (Assists)
  - xA (Expected Assists)
  - KP (Key Passes)
  - CrsPA (Crosses into Penalty Area)
  - PrgP (Progressive Passes)
  - Cmp% (Completed Passes Total)
  - PPA (Passes into Penalty Area)
  - Succ% (Successful Take-On%)
  - Carries
  - TklW (Tackles Won)
  - Tkl% (% of Dribblers Tackled)
  - Blocks
  - Pass (Passes Block)
  - Int (Interceptions)
  - Clr (Clearances)
  - Recov (Ball Recoveries)
  - Won% (% of Aerial Duels Won)
  - GA90 (Goals Against /90)
  - Save% (Save Percentage)
  - CS% (Clean Sheet Percentage)
  - Save%2 (Penalty Kicks Saved %)
  - Stp% (Crosses Stopped)
  - Launch% (% of Passes that were Launched)

**Response:**
```json
{
    "position": "string",
    "team": "string",
    "specific_role_cols": ["string"],
    "team_players": ["string"],
    "similar_players": [
        {
            "player": "string",
            "position": "string",
            "team": "string",
            "distance": float,
            "stats": {
                "attribute": float
            }
        }
    ]
}
```

**Example Request:**
```json
{
    "position": "Forward",
    "team": "Liverpool",
    "specific_role_cols": ["Gls", "SoT"]
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Successful request
- 400: Invalid input parameters
- 404: Team not found
- 500: Internal server error

## Interactive Documentation

FastAPI provides automatic interactive documentation. You can access it at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc` 