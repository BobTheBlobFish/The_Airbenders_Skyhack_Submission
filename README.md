
# Project Title

A brief description of what this project does and who it's for


## Problem Statement & Objective
Frontline teams at United Airlines are responsible for ensuring every flight departs on time and is operationally ready. However, not all flights are equally easy to manage. Certain flights pose greater complexity due to factors such as limited ground time, higher volumes of checked or carry-on baggage, and specific customer service needs that often increase with passenger load.
Currently, identifying these high-difficulty flights relies heavily on personal experience and local team knowledge. This manual approach is inconsistent, non-scalable, and risks missing opportunities for proactive resource planning across the airport.
To address this, you are tasked with developing a Flight Difficulty Score that systematically quantifies the relative complexity of each flight using the datasets provided, which span two weeks of departures from Chicago O’Hare International Airport (ORD).

The goal is to design a data-driven framework that:

- Calculates a Flight Difficulty Score for each flight using flight-level, customer, and station-level data.
- Identifies the primary operational drivers contributing to flight difficulty to enable proactive planning and optimized resource allocation.
## Approach Overview

Our solution is structured into three main stages:

### 1. Exploratory Data Analysis (EDA)
We examined two weeks of ORD flight data to uncover operational patterns:
- Average delay and % late departures  
- Flights with ground time below minimum turn threshold  
- Transfer-to-checked bag ratios  
- Load factor vs. delay correlation  
- Relationship between special service requests and delays  

### 2. Flight Difficulty Score Development
We built a **data-driven scoring model** that resets daily:
- **Feature categories:**
  - Ground time shortfall  
  - Passenger load & service needs  
  - Baggage handling complexity  
  - Flight- and aircraft-level attributes
- **Scoring:** Each flight is assigned a daily normalized score (0–100)  
- **Ranking:** Flights are ranked by difficulty per day  
- **Classification:** Flights grouped into:
  - 🟥 Difficult (Top 33%)  
  - 🟨 Medium (Middle 34%)  
  - 🟩 Easy (Bottom 33%)

### 3. Operational Insights
We analyzed the most common causes of operational complexity and identified airports and routes that consistently pose higher challenges.  
Key recommendations include optimizing staffing for tight-turn flights, prioritizing high-SSR flights and proactive baggage planning.

## Project Structure

```bash
The_Airbenders_Skyhack_Submission/
│
├── Question A/                         # Exploratory Data Analysis (EDA)
│   ├── eda_analysis_simple.py           # Script for analyzing flight delays, load factors, etc.
│   ├── eda_results.txt                  # Text summary of EDA findings
│   ├── README.md                        # Instructions and results for Question A
│   └── __pycache__/                     # Cached Python files
│
├── Question B/                         # Flight Difficulty Score Development
│   ├── flight_difficulty_scorer.py      # Core script for calculating difficulty scores
│   ├── daily_difficulty_summary.csv     # Summary of daily average difficulty scores
│   ├── flight_difficulty_scores.csv     # Flight-level difficulty scores and classifications
│   ├── top_difficult_flights.csv        # Ranked list of highest-difficulty flights
│   └── README.md                        # Explanation and methodology for scoring model
│
├── Question C/                         # Post-Analysis and Operational Insights
│   ├── operational_insights_analyzer.py # Script to derive insights and visualize results
│   ├── operational_insights_report.txt  # Summary of insights and key findings
│   ├── destination_difficulty_analysis.png     # Visualization of top difficult destinations
│   ├── destination_characteristics_heatmap.png # Heatmap of route difficulty
│   ├── difficulty_distribution.png             # Distribution of difficulty scores
│   ├── difficulty_driver_analysis.png          # Impact of different complexity factors
│   └── README.md                        # Overview and conclusions for Question C
│
├── data/                               # Raw input datasets 
├── SkyHack-30-2025-Submission.pdf       #pdf with all the results and findings
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Files to ignore in Git tracking
└── README.md                           # Main documentation 
```
## Scoring Steps

1. Normalize all features (Min-Max 0–1 scaling).  
2. Apply category weights to create composite difficulty score.  
3. Scale final score to **0–100** for interpretability.  
4. Rank flights *within each day* (highest = most difficult).  
5. Classify into **Difficult / Medium / Easy** using percentile thresholds.
## Results
All results are stored in **SkyHack-30-2025-Submission.pdf** that includes our findings, calculations and insights.
- **Score range:** 0.47 – 52.34  
- **Average score:** 16.77  
- **Classification distribution:**  
  - 🟥 Difficult: 33.4%  
  - 🟨 Medium: 33.4%  
  - 🟩 Easy: 33.2%---

## Exploratory Data Analysis (EDA) (from SkyHack-30-2025-Submission.pdf)

| Analysis | Key Finding |
|-----------|--------------|
| ✈️ **Flight Delays** | Avg. delay = **21.2 min**; 26.7% flights departed >15 min late |
| 🕒 **Ground Time** | 16.1% flights had <20% buffer; 7.8% below minimum turn time |
| 💼 **Baggage** | 60.1% transfer vs 39.9% checked; Avg. 78.8 bags/flight |
| 👥 **Passenger Load** | Avg. load factor = **102.4%**; 88.1% flights >80% load |
| ♿ **Special Services** | Avg. 48.7 SSR/flight; Wheelchairs = 45K+ of total 51K SSRs |

---
## Deployment

1. Clone the Repository and download all required dependencies:

```bash
  git clone https://github.com/BobTheBlobFish/The_Airbenders_Skyhack_Submission.git
  cd The_Airbenders_Skyhack_Submission
  pip install -r requirements.txt

```
2. For Question A: Exploratory Data Analysis (EDA):
```bash
  cd Question A
  python eda_analysis_simple.py
```
This updates the file eda_results.txt with questions and answers of all the sub-questions asked in the EDA Question

3. For Question B: Flight Difficulty Score Development:
```bash
  cd ..
  cd Question B
  python flight_difficulty_scorer.py
```
This updates the file top_difficult_flights.csv, flight_difficulty_scores.csv and daily_difficulty_summary.csv with questions and answers of all the sub-questions asked in the Flight Difficulty Score Development Question

4. For Question C: Post-Analysis & Operational Insights:
```bash
cd ..
  cd Question C
  python operational_insights_analyzer.py
```
This updates the file operational_insights_report.txt with questions and answers of all the sub-questions asked in the Post-Analysis & Operational Insights. The following files include visualizations:

a) destination_difficulty_analysis.png includes Visualization of top difficult destinations

b) destination_characteristics_heatmap.png includes Heatmap of route difficulty

c) difficulty_distribution.png includes Distribution of difficulty scores

d) difficulty_driver_analysis.png includes Impact of different complexity factors

## Destination Difficulty Analysis (from SkyHack-30-2025-Submission.pdf)

**Top 5 Difficult Destinations:**

| Rank | Airport | Difficulty % | Avg Score | Flights |
|------|----------|--------------|------------|----------|
| 1️⃣ | **GRU** (São Paulo) | 100% | 37.16 | 15 |
| 2️⃣ | **BRU** (Brussels) | 100% | 37.08 | 15 |
| 3️⃣ | **FRA** (Frankfurt) | 100% | 31.47 | 30 |
| 4️⃣ | **HNL** (Honolulu) | 100% | 31.44 | 15 |
| 5️⃣ | **ATH** (Athens) | 100% | 31.35 | 15 |

**Additional High-Difficulty Destinations:**  
HND (Tokyo), CDG (Paris), PUJ (Punta Cana), ORF (Norfolk), ONT (Ontario)


🗝️ *Pattern:*  
- International and long-haul routes show consistently higher operational complexity.  
- Domestic high-tourist routes (e.g., HNL, ORF) also face elevated ground-time and service challenges.
## Key Insights (from SkyHack-30-2025-Submission.pdf)

- ~38% of flights departed later than scheduled.  
- Flights with <10 min buffer below minimum turn were **2× more likely** to be delayed.  
- **Baggage volume** and **transfer ratio** were the **top complexity drivers** (2–3× increase in difficult flights).  
- **Special service requests (SSRs)** strongly correlated with higher difficulty.  
- “Difficult” flights accounted for **65%+ of total delay minutes**.
## Authors

### Team: The Airbenders

Members:

 Atharva Jakhetiya (2K22/EC/063)

 Akshit Sahu (2K22/EC/027)

