# United Airlines Flight Difficulty Score - Exploratory Data Analysis

## Overview
This folder contains the complete Exploratory Data Analysis (EDA) for the United Airlines Flight Difficulty Score project. The analysis addresses all 5 required EDA questions using data from Chicago O'Hare International Airport (ORD).

## Files
- `eda_analysis.py`: Complete EDA implementation with visualizations (requires matplotlib/plotly)
- `eda_analysis_simple.py`: Simplified EDA implementation without complex visualizations
- `test_eda.py`: Test script to verify implementation
- `requirements.txt`: Python dependencies
- `README.md`: This file with instructions

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the simplified EDA analysis (recommended):
```bash
python eda_analysis_simple.py
```

3. Or run the full EDA with visualizations:
```bash
python eda_analysis.py
```

## EDA Questions Addressed
1. **Delay Analysis**: What is the average delay and what percentage of flights depart later than scheduled?
2. **Ground Time Analysis**: How many flights have scheduled ground time close to or below the minimum turn mins?
3. **Baggage Analysis**: What is the average ratio of transfer bags vs. checked bags across flights?
4. **Passenger Load Analysis**: How do passenger loads compare across flights, and do higher loads correlate with operational difficulty?
5. **Special Services Analysis**: Are high special service requests flights also high-delay after controlling for load?

## Implementation Features
- **Comprehensive Data Processing**: Handles all 5 datasets with proper data cleaning and preprocessing
- **Statistical Analysis**: Calculates key metrics, correlations, and distributions
- **Flexible Visualization**: Optional matplotlib/plotly support for charts and graphs
- **Robust Error Handling**: Graceful handling of missing dependencies and data issues
- **Detailed Reporting**: Comprehensive output with statistics and insights

## Output
The analysis generates:
- Statistical summaries for each EDA question
- Key findings and insights
- Data quality metrics
- Operational complexity indicators

## Data Requirements
Ensure the following files are in the `../data/` directory:
- `Flight Level Data.csv`
- `PNR+Flight+Level+Data.csv`
- `PNR Remark Level Data.csv`
- `Bag+Level+Data.csv`
- `Airports Data.csv`

## Notes
- The simplified version (`eda_analysis_simple.py`) is recommended for environments without visualization libraries
- Both versions provide the same core analysis and statistical insights
- All calculations are performed using pandas and numpy for reliability

