# United Airlines Flight Difficulty Score Development

## Overview
This folder contains the implementation of a systematic daily-level Flight Difficulty Score that ranks and classifies flights based on operational complexity factors.

## Files
- `flight_difficulty_scorer.py`: Main implementation of the difficulty scoring system
- `requirements.txt`: Python dependencies
- `README.md`: This documentation file

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flight Difficulty Scoring:
```bash
python flight_difficulty_scorer.py
```

## Methodology

### Feature Engineering
The system calculates difficulty scores based on multiple operational factors:

#### 1. Ground Time Complexity (35% weight)
- **Ground Time Pressure**: Inverse of ground time ratio (lower ratio = higher difficulty)
- **Ground Time Risk**: Penalty for negative buffer time, reward for positive buffer

#### 2. Aircraft Complexity (15% weight)
- **Aircraft Size**: Larger aircraft require more complex operations
- **Carrier Type**: Mainline vs Express operations complexity

#### 3. Passenger Complexity (25% weight)
- **Load Factor**: Passenger capacity utilization
- **Children Complexity**: Ratio of children/infants per passenger
- **Stroller Complexity**: Ratio of stroller users per passenger
- **Basic Economy Complexity**: Ratio of basic economy passengers

#### 4. Special Services Complexity (10% weight)
- **Service Requests**: Special assistance requests per passenger
- **Service Types**: Wheelchair assistance, unaccompanied minors, etc.

#### 5. Baggage Complexity (10% weight)
- **Baggage Volume**: Bags per passenger ratio
- **Transfer Bag Complexity**: Transfer bags require more coordination

#### 6. Historical Performance (5% weight)
- **Delay History**: Past performance as difficulty indicator

### Scoring Process

1. **Feature Normalization**: All features normalized to 0-1 scale
2. **Weighted Scoring**: Features combined using domain-expertise weights
3. **Score Scaling**: Final scores scaled to 0-100 for interpretation

### Daily Ranking System

#### Ranking Methodology:
- **Daily Reset**: Rankings calculated fresh for each day
- **Score-Based Ranking**: Flights ranked by difficulty score within each day
- **Dense Ranking**: Handles ties appropriately

#### Classification System:
- **Difficult**: Top 33% of flights by difficulty score
- **Medium**: Middle 33% of flights by difficulty score  
- **Easy**: Bottom 33% of flights by difficulty score

## Output Files

### 1. `flight_difficulty_scores.csv`
Complete dataset with all flights, features, scores, rankings, and classifications.

### 2. `daily_difficulty_summary.csv`
Daily summary statistics including:
- Average, min, max difficulty scores
- Standard deviation
- Flight counts
- Classification distributions

### 3. `top_difficult_flights.csv`
Top 50 most difficult flights across all days for operational planning.

## Key Features

- **Daily-Level Analysis**: Rankings reset each day for fair comparison
- **Multi-Factor Scoring**: Considers all major operational complexity drivers
- **Actionable Classifications**: Three-tier system for resource allocation
- **Comprehensive Reporting**: Detailed statistics and summaries
- **Scalable Design**: Easy to adjust weights and add new features

## Operational Applications

### Resource Planning:
- **Difficult flights**: Allocate additional ground crew, gate agents
- **Medium flights**: Standard resource allocation
- **Easy flights**: Minimal resource allocation

### Proactive Management:
- **Early identification** of challenging flights
- **Resource optimization** based on difficulty levels
- **Performance monitoring** through daily rankings

### Continuous Improvement:
- **Weight tuning** based on operational feedback
- **Feature addition** as new complexity factors emerge
- **Performance validation** against actual operational outcomes

## Technical Notes

- **Memory Efficient**: Handles large datasets with chunked processing
- **Robust Error Handling**: Graceful handling of missing data
- **Scalable Architecture**: Easy to extend with new features
- **Performance Optimized**: Efficient pandas operations for large datasets

