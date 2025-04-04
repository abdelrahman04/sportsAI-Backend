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
