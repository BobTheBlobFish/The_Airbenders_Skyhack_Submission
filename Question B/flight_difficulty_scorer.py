#!/usr/bin/env python3
"""
United Airlines Flight Difficulty Score Development
=================================================

This script implements a systematic daily-level scoring approach that:
1. Calculates Flight Difficulty Scores for each flight
2. Ranks flights within each day by difficulty
3. Classifies flights into Difficult/Medium/Easy categories

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class FlightDifficultyScorer:
    """Class to calculate and manage Flight Difficulty Scores"""
    
    def __init__(self, data_path="../data/"):
        """Initialize with data path"""
        self.data_path = data_path
        self.flights = None
        self.pnr_flight = None
        self.pnr_remarks = None
        self.bags = None
        self.airports = None
        self.flight_features = None
        
    def load_data(self):
        """Load all datasets"""
        print("Loading datasets for Flight Difficulty Scoring...")
        
        # Load flight level data
        self.flights = pd.read_csv(f"{self.data_path}Flight Level Data.csv")
        print(f" Loaded {len(self.flights):,} flight records")
        
        # Load PNR flight level data (in chunks to handle large file)
        print("Loading PNR flight data (large file, please wait)...")
        chunk_list = []
        chunk_size = 100000
        for chunk in pd.read_csv(f"{self.data_path}PNR+Flight+Level+Data.csv", chunksize=chunk_size):
            chunk_list.append(chunk)
        self.pnr_flight = pd.concat(chunk_list, ignore_index=True)
        print(f" Loaded {len(self.pnr_flight):,} PNR flight records")
        
        # Load PNR remarks data
        self.pnr_remarks = pd.read_csv(f"{self.data_path}PNR Remark Level Data.csv")
        print(f" Loaded {len(self.pnr_remarks):,} PNR remark records")
        
        # Load bag level data (in chunks to handle large file)
        print("Loading bag data (large file, please wait)...")
        chunk_list = []
        for chunk in pd.read_csv(f"{self.data_path}Bag+Level+Data.csv", chunksize=chunk_size):
            chunk_list.append(chunk)
        self.bags = pd.concat(chunk_list, ignore_index=True)
        print(f" Loaded {len(self.bags):,} bag records")
        
        # Load airports data
        self.airports = pd.read_csv(f"{self.data_path}Airports Data.csv")
        print(f" Loaded {len(self.airports):,} airport records")
        
        print("\nData loading completed successfully!")
        
    def preprocess_data(self):
        """Preprocess and clean the data"""
        print("\nPreprocessing data...")
        
        # Convert datetime columns
        datetime_cols = [
            'scheduled_departure_datetime_local',
            'scheduled_arrival_datetime_local', 
            'actual_departure_datetime_local',
            'actual_arrival_datetime_local'
        ]
        
        for col in datetime_cols:
            if col in self.flights.columns:
                self.flights[col] = pd.to_datetime(self.flights[col])
        
        # Convert date columns
        date_cols = ['scheduled_departure_date_local', 'pnr_creation_date', 'bag_tag_issue_date']
        for col in date_cols:
            if col in self.flights.columns:
                self.flights[col] = pd.to_datetime(self.flights[col])
            if col in self.pnr_flight.columns:
                self.pnr_flight[col] = pd.to_datetime(self.pnr_flight[col])
            if col in self.bags.columns:
                self.bags[col] = pd.to_datetime(self.bags[col])
        
        # Calculate delays
        self.flights['departure_delay_minutes'] = (
            self.flights['actual_departure_datetime_local'] - 
            self.flights['scheduled_departure_datetime_local']
        ).dt.total_seconds() / 60
        
        self.flights['arrival_delay_minutes'] = (
            self.flights['actual_arrival_datetime_local'] - 
            self.flights['scheduled_arrival_datetime_local']
        ).dt.total_seconds() / 60
        
        # Calculate ground time ratios
        # Fix: Handle division by zero in minimum_turn_minutes
        self.flights['ground_time_ratio'] = np.where(
            self.flights['minimum_turn_minutes'] > 0,
            self.flights['scheduled_ground_time_minutes'] / self.flights['minimum_turn_minutes'],
            0.01  # Very low ratio for flights with 0 minimum turn time
        )
        
        self.flights['ground_time_buffer'] = (
            self.flights['scheduled_ground_time_minutes'] - 
            self.flights['minimum_turn_minutes']
        )
        
        print(" Data preprocessing completed!")
        
    def engineer_features(self):
        """Engineer features for flight difficulty scoring"""
        print("\nEngineering features for difficulty scoring...")
        
        # Start with flight data
        self.flight_features = self.flights.copy()
        
        # 1. GROUND TIME COMPLEXITY FEATURES
        print(" Calculating ground time complexity features...")
        
        # Ground time pressure (lower ratio = more difficult)
        # Fix: Handle division by zero and very small ratios to prevent infinity
        self.flight_features['ground_time_pressure'] = np.where(
            self.flight_features['ground_time_ratio'] > 0.01,  # Avoid division by very small numbers
            1 / self.flight_features['ground_time_ratio'],
            100  # High penalty for flights with very low ground time ratios
        )
        
        # Ground time buffer risk (negative buffer = high risk)
        self.flight_features['ground_time_risk'] = np.where(
            self.flight_features['ground_time_buffer'] < 0, 
            abs(self.flight_features['ground_time_buffer']) + 10,  # Penalty for negative buffer
            np.maximum(0, 10 - self.flight_features['ground_time_buffer'])  # Reward for positive buffer
        )
        
        # 2. AIRCRAFT COMPLEXITY FEATURES
        print(" Calculating aircraft complexity features...")
        
        # Aircraft size complexity (larger aircraft = more complex)
        self.flight_features['aircraft_size_complexity'] = self.flight_features['total_seats'] / 100
        
        # Fleet type complexity (Mainline vs Express)
        self.flight_features['carrier_complexity'] = np.where(
            self.flight_features['carrier'] == 'Mainline', 1.5, 1.0
        )
        
        # 3. PASSENGER COMPLEXITY FEATURES
        print(" Calculating passenger complexity features...")
        
        # Convert string columns to numeric BEFORE aggregation
        self.pnr_flight['is_child'] = pd.to_numeric(self.pnr_flight['is_child'], errors='coerce').fillna(0)
        self.pnr_flight['basic_economy_ind'] = pd.to_numeric(self.pnr_flight['basic_economy_ind'], errors='coerce').fillna(0)
        # Fix: Properly convert Y/N strings to 1/0 for stroller users
        self.pnr_flight['is_stroller_user'] = self.pnr_flight['is_stroller_user'].map({'Y': 1, 'N': 0}).fillna(0)
        
        # Aggregate PNR data by flight
        pnr_summary = self.pnr_flight.groupby([
            'company_id', 'flight_number', 'scheduled_departure_date_local'
        ]).agg({
            'total_pax': 'sum',
            'lap_child_count': 'sum',
            'is_child': 'sum',
            'basic_economy_ind': 'sum',
            'is_stroller_user': 'sum'
        }).reset_index()
        
        # No need to convert again since we already did it above
        
        # Merge with flight data
        self.flight_features = self.flight_features.merge(
            pnr_summary,
            on=['company_id', 'flight_number', 'scheduled_departure_date_local'],
            how='left'
        )
        
        # Fill missing values
        self.flight_features['total_pax'] = self.flight_features['total_pax'].fillna(0)
        self.flight_features['lap_child_count'] = self.flight_features['lap_child_count'].fillna(0)
        self.flight_features['is_child'] = self.flight_features['is_child'].fillna(0)
        self.flight_features['basic_economy_ind'] = self.flight_features['basic_economy_ind'].fillna(0)
        self.flight_features['is_stroller_user'] = self.flight_features['is_stroller_user'].fillna(0)
        
        # Calculate passenger complexity metrics
        # Fix: Handle division by zero in total_seats
        self.flight_features['load_factor'] = np.where(
            self.flight_features['total_seats'] > 0,
            self.flight_features['total_pax'] / self.flight_features['total_seats'],
            0  # No load factor if no seats
        )
        
        self.flight_features['children_complexity'] = (
            self.flight_features['lap_child_count'] + self.flight_features['is_child']
        ) / self.flight_features['total_pax'].replace(0, np.nan)
        self.flight_features['children_complexity'] = self.flight_features['children_complexity'].fillna(0)
        
        self.flight_features['stroller_complexity'] = (
            self.flight_features['is_stroller_user'] / self.flight_features['total_pax'].replace(0, np.nan)
        )
        self.flight_features['stroller_complexity'] = self.flight_features['stroller_complexity'].fillna(0)
        
        self.flight_features['basic_economy_complexity'] = (
            self.flight_features['basic_economy_ind'] / self.flight_features['total_pax'].replace(0, np.nan)
        )
        self.flight_features['basic_economy_complexity'] = self.flight_features['basic_economy_complexity'].fillna(0)
        
        # 4. SPECIAL SERVICES COMPLEXITY FEATURES
        print(" Calculating special services complexity features...")
        
        # Count special services per flight
        special_services = self.pnr_remarks.groupby([
            'flight_number'
        ]).size().reset_index(name='special_service_count')
        
        # Merge with flight data
        self.flight_features = self.flight_features.merge(
            special_services,
            on='flight_number',
            how='left'
        )
        self.flight_features['special_service_count'] = self.flight_features['special_service_count'].fillna(0)
        
        # Calculate special service complexity (absolute count, not per-passenger ratio)
        # Special services require the same operational effort regardless of passenger count
        self.flight_features['special_service_complexity'] = self.flight_features['special_service_count']
        
        # 5. BAGGAGE COMPLEXITY FEATURES
        print(" Calculating baggage complexity features...")
        
        # Count bags by type per flight
        bag_counts = self.bags.groupby([
            'company_id', 'flight_number', 'scheduled_departure_date_local', 'bag_type'
        ]).size().unstack(fill_value=0).reset_index()
        
        # Calculate ratios (using 'Origin' instead of 'Checked' as per data dictionary)
        bag_counts['total_bags'] = bag_counts.get('Origin', 0) + bag_counts.get('Transfer', 0) + bag_counts.get('Hot Transfer', 0)
        bag_counts['transfer_ratio'] = (bag_counts.get('Transfer', 0) + bag_counts.get('Hot Transfer', 0)) / bag_counts['total_bags'].replace(0, np.nan)
        bag_counts['transfer_ratio'] = bag_counts['transfer_ratio'].fillna(0)
        
        # Merge with flight data
        self.flight_features = self.flight_features.merge(
            bag_counts, 
            on=['company_id', 'flight_number', 'scheduled_departure_date_local'],
            how='left'
        )
        
        # Fill missing values
        self.flight_features['total_bags'] = self.flight_features['total_bags'].fillna(0)
        self.flight_features['transfer_ratio'] = self.flight_features['transfer_ratio'].fillna(0)
        
        # Calculate baggage complexity (absolute count, not per-passenger ratio)
        # Baggage handling complexity is about total volume, not per-passenger ratios
        self.flight_features['baggage_complexity'] = self.flight_features['total_bags']
        
        # Transfer bags are more complex to handle - use absolute count with complexity multiplier
        transfer_bags = self.flight_features.get('Transfer', 0).fillna(0) + self.flight_features.get('Hot Transfer', 0).fillna(0)
        self.flight_features['transfer_bag_complexity'] = transfer_bags * 2  # Transfer bags are 2x more complex
        
        # 6. HISTORICAL PERFORMANCE FEATURES
        print(" Calculating historical performance features...")
        
        # Use departure delay as a proxy for historical difficulty
        self.flight_features['historical_difficulty'] = np.where(
            self.flight_features['departure_delay_minutes'] > 15, 1.5,  # High delay penalty
            np.where(self.flight_features['departure_delay_minutes'] > 0, 1.2, 1.0)  # Moderate delay penalty
        )
        
        print(" Feature engineering completed!")
        
    def calculate_difficulty_score(self):
        """Calculate the composite Flight Difficulty Score"""
        print("\nCalculating Flight Difficulty Scores...")
        
        # Define feature weights (these can be tuned based on domain expertise)
        weights = {
            'ground_time_pressure': 0.20,
            'ground_time_risk': 0.15,
            'aircraft_size_complexity': 0.10,
            'carrier_complexity': 0.05,
            'load_factor': 0.10,
            'children_complexity': 0.10,
            'stroller_complexity': 0.05,
            'basic_economy_complexity': 0.05,
            'special_service_complexity': 0.10,
            'baggage_complexity': 0.05,
            'transfer_bag_complexity': 0.05
        }
        
        # Normalize features to 0-1 scale
        normalized_features = {}
        for feature in weights.keys():
            if feature in self.flight_features.columns:
                # Min-max normalization
                min_val = self.flight_features[feature].min()
                max_val = self.flight_features[feature].max()
                if max_val > min_val:
                    normalized_features[feature] = (
                        self.flight_features[feature] - min_val
                    ) / (max_val - min_val)
                else:
                    normalized_features[feature] = 0
        
        # Calculate weighted composite score
        self.flight_features['difficulty_score'] = 0
        for feature, weight in weights.items():
            if feature in normalized_features:
                self.flight_features['difficulty_score'] += (
                    normalized_features[feature] * weight
                )
        
        # Scale to 0-100 for easier interpretation
        self.flight_features['difficulty_score'] = self.flight_features['difficulty_score'] * 100
        
        print(f" Difficulty scores calculated for {len(self.flight_features):,} flights")
        print(f" Score range: {self.flight_features['difficulty_score'].min():.2f} - {self.flight_features['difficulty_score'].max():.2f}")
        
    def create_daily_rankings(self):
        """Create daily rankings and classifications"""
        print("\nCreating daily rankings and classifications...")
        
        # Sort by date and difficulty score
        self.flight_features = self.flight_features.sort_values([
            'scheduled_departure_date_local', 'difficulty_score'
        ], ascending=[True, False])  # False for descending difficulty
        
        # Create daily rankings
        self.flight_features['daily_rank'] = self.flight_features.groupby(
            'scheduled_departure_date_local'
        )['difficulty_score'].rank(method='dense', ascending=False)
        
        # Calculate total flights per day for percentile calculation
        daily_counts = self.flight_features.groupby('scheduled_departure_date_local').size()
        self.flight_features['daily_flight_count'] = self.flight_features['scheduled_departure_date_local'].map(daily_counts)
        
        # Calculate percentile rank within each day
        self.flight_features['daily_percentile'] = (
            self.flight_features['daily_rank'] / self.flight_features['daily_flight_count']
        ) * 100
        
        # Create three-tier classification
        def classify_difficulty(row):
            if row['daily_percentile'] <= 33.33:
                return 'Difficult'
            elif row['daily_percentile'] <= 66.67:
                return 'Medium'
            else:
                return 'Easy'
        
        self.flight_features['difficulty_classification'] = self.flight_features.apply(classify_difficulty, axis=1)
        
        print(" Daily rankings and classifications completed!")
        
    def generate_summary_report(self):
        """Generate summary report of the difficulty scoring"""
        print("\n" + "="*60)
        print("FLIGHT DIFFICULTY SCORE SUMMARY REPORT")
        print("="*60)
        
        # Overall statistics
        print(f"\nOVERALL STATISTICS:")
        print(f" Total flights analyzed: {len(self.flight_features):,}")
        print(f" Date range: {self.flight_features['scheduled_departure_date_local'].min().strftime('%Y-%m-%d')} to {self.flight_features['scheduled_departure_date_local'].max().strftime('%Y-%m-%d')}")
        print(f" Average difficulty score: {self.flight_features['difficulty_score'].mean():.2f}")
        print(f" Median difficulty score: {self.flight_features['difficulty_score'].median():.2f}")
        
        # Classification distribution
        print(f"\nCLASSIFICATION DISTRIBUTION:")
        classification_counts = self.flight_features['difficulty_classification'].value_counts()
        for classification, count in classification_counts.items():
            percentage = (count / len(self.flight_features)) * 100
            print(f" {classification}: {count:,} flights ({percentage:.1f}%)")
        
        # Daily breakdown
        print(f"\nDAILY BREAKDOWN:")
        daily_summary = self.flight_features.groupby('scheduled_departure_date_local').agg({
            'difficulty_score': ['mean', 'std', 'min', 'max'],
            'difficulty_classification': lambda x: x.value_counts().to_dict()
        }).round(2)
        
        for date in sorted(self.flight_features['scheduled_departure_date_local'].unique()):
            day_data = self.flight_features[self.flight_features['scheduled_departure_date_local'] == date]
            print(f" {date.strftime('%Y-%m-%d')}: {len(day_data):,} flights")
            print(f"   Score range: {day_data['difficulty_score'].min():.1f} - {day_data['difficulty_score'].max():.1f}")
            print(f"   Classification: {day_data['difficulty_classification'].value_counts().to_dict()}")
        
        # Top 10 most difficult flights
        print(f"\nTOP 10 MOST DIFFICULT FLIGHTS:")
        top_difficult = self.flight_features.nlargest(10, 'difficulty_score')[
            ['company_id', 'flight_number', 'scheduled_departure_date_local', 
             'scheduled_arrival_station_code', 'difficulty_score', 'difficulty_classification']
        ]
        for idx, row in top_difficult.iterrows():
            print(f" {row['company_id']}{row['flight_number']} to {row['scheduled_arrival_station_code']} on {row['scheduled_departure_date_local'].strftime('%Y-%m-%d')}: {row['difficulty_score']:.1f} ({row['difficulty_classification']})")
        
        print(f"\n Flight Difficulty Scoring completed successfully!")
        
    def save_results(self):
        """Save the results to CSV files"""
        print("\nSaving results...")
        
        # Save full results
        output_file = "flight_difficulty_scores.csv"
        self.flight_features.to_csv(output_file, index=False)
        print(f" Full results saved to: {output_file}")
        
        # Save summary by date
        daily_summary = self.flight_features.groupby('scheduled_departure_date_local').agg({
            'difficulty_score': ['mean', 'std', 'min', 'max', 'count'],
            'difficulty_classification': lambda x: x.value_counts().to_dict()
        }).round(2)
        
        daily_summary.columns = ['avg_score', 'std_score', 'min_score', 'max_score', 'flight_count', 'classification_distribution']
        daily_summary.to_csv("daily_difficulty_summary.csv")
        print(f" Daily summary saved to: daily_difficulty_summary.csv")
        
        # Save top difficult flights
        top_difficult = self.flight_features.nlargest(50, 'difficulty_score')
        top_difficult.to_csv("top_difficult_flights.csv", index=False)
        print(f" Top 50 difficult flights saved to: top_difficult_flights.csv")
        
    def run_complete_scoring(self):
        """Run the complete Flight Difficulty Scoring process"""
        print("UNITED AIRLINES FLIGHT DIFFICULTY SCORE DEVELOPMENT")
        print("="*60)
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Engineer features
        self.engineer_features()
        
        # Calculate difficulty scores
        self.calculate_difficulty_score()
        
        # Create daily rankings and classifications
        self.create_daily_rankings()
        
        # Generate summary report
        self.generate_summary_report()
        
        # Save results
        self.save_results()
        
        return self.flight_features

def main():
    """Main function to run the Flight Difficulty Scoring"""
    # Initialize scorer
    scorer = FlightDifficultyScorer(data_path="../data/")
    
    # Run complete scoring process
    results = scorer.run_complete_scoring()
    
    return results

if __name__ == "__main__":
    results = main()
