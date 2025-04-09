# SportsAI - Football Player Analysis API

A FastAPI-based backend service for analyzing football players based on their positions, teams, and specific role attributes.

## Features

- Player analysis based on position and team
- Customizable role-specific attributes for analysis
- Similar player recommendations
- Support for different player positions including:
  - Forward
  - Winger
  - Attacking Mid
  - Centre Mid
  - Fullback
  - Centre Defense
  - Goalkeeping

## API Endpoints

### POST /analyze-players

Analyzes players based on position, team, and specific role attributes.

Request body:
```json
{
    "position": "string",
    "team": "string",
    "specific_role_cols": ["string"]
}
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python api.py
```

The server will start on http://localhost:8000

## Data

The analysis uses two main data sources:
- Players Merged.csv: Contains data for outfield players
- Players GK Merged.csv: Contains data for goalkeepers

## Data Processing Steps

The data used in this analysis goes through several processing steps:

1. **Data Collection**:
   - Extracting data from FBref using Excel exports
   - Web scraping to obtain exact player positions from Transfermarkt

2. **Data Cleaning**:
   - Handling data types and converting to appropriate formats
   - Processing irregular rows (Rk column)
   - Standardizing league and team columns
   - Cleaning players with unknown, missing, or incorrect positions

3. **Dataset Integration**:
   - Combining multiple datasets to create two major player datasets:
     - Players Merged.csv: For outfield players
     - Players GK Merged.csv: For goalkeepers
   - Cleaning team dataset data types
   - Integrating team datasets to create two major team datasets:
     - Teams dataset
     - Teams GK dataset
