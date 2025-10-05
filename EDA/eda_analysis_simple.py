#!/usr/bin/env python3
"""
United Airlines Flight Difficulty Score - Exploratory Data Analysis
================================================================

This script performs comprehensive EDA on flight data from Chicago O'Hare International Airport (ORD)
to address the 5 required analysis questions for the Flight Difficulty Score project.

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class FlightEDA:
    """Class to handle all EDA operations for flight difficulty analysis"""
    
    def __init__(self, data_path="../data/"):
        """Initialize with data path"""
        self.data_path = data_path
        self.flights = None
        self.pnr_flight = None
        self.pnr_remarks = None
        self.bags = None
        self.airports = None
        
    def load_data(self):
        """Load all datasets"""
        print("Loading datasets...")
        
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
        self.flights['ground_time_ratio'] = (
            self.flights['scheduled_ground_time_minutes'] / 
            self.flights['minimum_turn_minutes']
        )
        
        self.flights['ground_time_buffer'] = (
            self.flights['scheduled_ground_time_minutes'] - 
            self.flights['minimum_turn_minutes']
        )
        
        print(" Data preprocessing completed!")
        
    def question_1_delay_analysis(self):
        """Question 1: Average delay and percentage of late departures"""
        print("\n" + "="*60)
        print("QUESTION 1: DELAY ANALYSIS")
        print("="*60)
        
        # Calculate statistics
        avg_delay = self.flights['departure_delay_minutes'].mean()
        median_delay = self.flights['departure_delay_minutes'].median()
        
        # Flights departing late (>15 minutes)
        late_flights = self.flights[self.flights['departure_delay_minutes'] > 15]
        late_percentage = (len(late_flights) / len(self.flights)) * 100
        
        # Flights departing early
        early_flights = self.flights[self.flights['departure_delay_minutes'] <= 15]
        early_percentage = (len(early_flights) / len(self.flights)) * 100
        
        print(f" DELAY STATISTICS:")
        print(f"    Average departure delay: {avg_delay:.1f} minutes")
        print(f"    Median departure delay: {median_delay:.1f} minutes")
        print(f"    Flights departing late (>15 min): {late_percentage:.1f}%")
        print(f"    Flights departing early (<=15 min): {early_percentage:.1f}%")
        
        
        return {
            'avg_delay': avg_delay,
            'median_delay': median_delay,
            'late_percentage': late_percentage,
            'early_percentage': early_percentage
        }
    
    def question_2_ground_time_analysis(self):
        """Question 2: Flights with tight ground time"""
        print("\n" + "="*60)
        print("QUESTION 2: GROUND TIME ANALYSIS")
        print("="*60)
        
        # Flights with tight ground time (ratio < 1.2)
        tight_ground_time = self.flights[self.flights['ground_time_ratio'] < 1.2]
        tight_percentage = (len(tight_ground_time) / len(self.flights)) * 100
        
        # Flights with ground time below minimum
        below_minimum = self.flights[self.flights['ground_time_ratio'] < 1.0]
        below_minimum_percentage = (len(below_minimum) / len(self.flights)) * 100
        
        # Average ground time buffer
        avg_buffer = self.flights['ground_time_buffer'].mean()
        
        print(f" GROUND TIME STATISTICS:")
        print(f"    Flights with tight ground time (<20% buffer): {tight_percentage:.1f}%")
        print(f"    Flights below minimum turn time: {below_minimum_percentage:.1f}%")
        print(f"    Average ground time buffer: {avg_buffer:.1f} minutes")
        
        # Additional analysis
        print(f"\n GROUND TIME DISTRIBUTION:")
        print(f"    Average ground time ratio: {self.flights['ground_time_ratio'].mean():.2f}")
        print(f"    Median ground time ratio: {self.flights['ground_time_ratio'].median():.2f}")
        print(f"    25th percentile ratio: {self.flights['ground_time_ratio'].quantile(0.25):.2f}")
        print(f"    75th percentile ratio: {self.flights['ground_time_ratio'].quantile(0.75):.2f}")
        
        return {
            'tight_ground_time_percentage': tight_percentage,
            'below_minimum_percentage': below_minimum_percentage,
            'avg_buffer': avg_buffer
        }
    
    def question_3_baggage_analysis(self):
        """Question 3: Transfer vs checked bag ratios"""
        print("\n" + "="*60)
        print("QUESTION 3: BAGGAGE ANALYSIS")
        print("="*60)
        
        # Count bags by type per flight
        bag_counts = self.bags.groupby([
            'company_id', 'flight_number', 'scheduled_departure_date_local', 'bag_type'
        ]).size().unstack(fill_value=0).reset_index()
        
        # Calculate ratios
        bag_counts['total_bags'] = bag_counts.get('Checked', 0) + bag_counts.get('Transfer', 0)
        bag_counts['transfer_ratio'] = bag_counts.get('Transfer', 0) / bag_counts['total_bags'].replace(0, np.nan)
        bag_counts['checked_ratio'] = bag_counts.get('Checked', 0) / bag_counts['total_bags'].replace(0, np.nan)
        
        # Merge with flight data
        flight_bags = self.flights.merge(
            bag_counts, 
            on=['company_id', 'flight_number', 'scheduled_departure_date_local'],
            how='left'
        )
        
        # Calculate statistics
        avg_transfer_ratio = flight_bags['transfer_ratio'].mean()
        avg_checked_ratio = flight_bags['checked_ratio'].mean()
        avg_bags_per_flight = flight_bags['total_bags'].mean()
        
        print(f" BAGGAGE STATISTICS:")
        print(f"    Average transfer bag ratio: {avg_transfer_ratio:.3f}")
        print(f"    Average checked bag ratio: {avg_checked_ratio:.3f}")
        print(f"    Average bags per flight: {avg_bags_per_flight:.1f}")
        
        # Additional analysis
        print(f"\n BAGGAGE DISTRIBUTION:")
        print(f"    Total bags in dataset: {len(self.bags):,}")
        print(f"    Transfer bags: {len(self.bags[self.bags['bag_type'] == 'Transfer']):,}")
        print(f"    Checked bags: {len(self.bags[self.bags['bag_type'] == 'Checked']):,}")
        print(f"    Flights with bag data: {len(flight_bags[flight_bags['total_bags'] > 0]):,}")
        
        return {
            'avg_transfer_ratio': avg_transfer_ratio,
            'avg_checked_ratio': avg_checked_ratio,
            'avg_bags_per_flight': avg_bags_per_flight
        }
    
    def question_4_passenger_load_analysis(self):
        """Question 4: Passenger loads and operational difficulty correlation"""
        print("\n" + "="*60)
        print("QUESTION 4: PASSENGER LOAD ANALYSIS")
        print("="*60)
        
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
        
        # Merge with flight data
        flight_passengers = self.flights.merge(
            pnr_summary,
            on=['company_id', 'flight_number', 'scheduled_departure_date_local'],
            how='left'
        )
        
        # Calculate load factor
        flight_passengers['load_factor'] = (
            flight_passengers['total_pax'] / flight_passengers['total_seats']
        )
        
        # Convert string columns to numeric for calculations
        flight_passengers['is_child'] = pd.to_numeric(flight_passengers['is_child'], errors='coerce').fillna(0)
        flight_passengers['basic_economy_ind'] = pd.to_numeric(flight_passengers['basic_economy_ind'], errors='coerce').fillna(0)
        flight_passengers['is_stroller_user'] = pd.to_numeric(flight_passengers['is_stroller_user'], errors='coerce').fillna(0)
        
        # Calculate passenger complexity metrics
        flight_passengers['children_ratio'] = (
            flight_passengers['lap_child_count'] + flight_passengers['is_child']
        ) / flight_passengers['total_pax'].replace(0, np.nan)
        
        flight_passengers['basic_economy_ratio'] = (
            flight_passengers['basic_economy_ind'] / flight_passengers['total_pax'].replace(0, np.nan)
        )
        
        flight_passengers['stroller_ratio'] = (
            flight_passengers['is_stroller_user'] / flight_passengers['total_pax'].replace(0, np.nan)
        )
        
        # Calculate statistics
        avg_load_factor = flight_passengers['load_factor'].mean()
        high_load_flights = flight_passengers[flight_passengers['load_factor'] > 0.8]
        high_load_percentage = (len(high_load_flights) / len(flight_passengers)) * 100
        
        # Correlation analysis
        load_delay_corr = flight_passengers['load_factor'].corr(flight_passengers['departure_delay_minutes'])
        
        print(f" PASSENGER LOAD STATISTICS:")
        print(f"    Average load factor: {avg_load_factor:.3f}")
        print(f"    High load flights (>80%): {high_load_percentage:.1f}%")
        print(f"    Load factor vs delay correlation: {load_delay_corr:.3f}")
        
        # Additional analysis
        print(f"\n PASSENGER COMPLEXITY:")
        print(f"    Average children ratio: {flight_passengers['children_ratio'].mean():.3f}")
        print(f"    Average basic economy ratio: {flight_passengers['basic_economy_ratio'].mean():.3f}")
        print(f"    Average stroller ratio: {flight_passengers['stroller_ratio'].mean():.3f}")
        print(f"    Total passengers in dataset: {flight_passengers['total_pax'].sum():,}")
        
        return {
            'avg_load_factor': avg_load_factor,
            'high_load_percentage': high_load_percentage,
            'load_delay_correlation': load_delay_corr
        }
    
    def question_5_special_services_analysis(self):
        """Question 5: Special service requests and delays"""
        print("\n" + "="*60)
        print("QUESTION 5: SPECIAL SERVICES ANALYSIS")
        print("="*60)
        
        # Count special services per flight
        special_services = self.pnr_remarks.groupby([
            'flight_number'
        ]).size().reset_index(name='special_service_count')
        
        # Merge with flight data
        flight_services = self.flights.merge(
            special_services,
            on='flight_number',
            how='left'
        )
        flight_services['special_service_count'] = flight_services['special_service_count'].fillna(0)
        
        # Merge with passenger data for load control
        pnr_summary = self.pnr_flight.groupby([
            'company_id', 'flight_number', 'scheduled_departure_date_local'
        ]).agg({
            'total_pax': 'sum'
        }).reset_index()
        
        flight_services = flight_services.merge(
            pnr_summary,
            on=['company_id', 'flight_number', 'scheduled_departure_date_local'],
            how='left'
        )
        flight_services['total_pax'] = flight_services['total_pax'].fillna(0)
        
        # Calculate special service ratio
        flight_services['special_service_ratio'] = (
            flight_services['special_service_count'] / flight_services['total_pax'].replace(0, np.nan)
        )
        
        # Categorize flights by special service intensity
        flight_services['service_category'] = pd.cut(
            flight_services['special_service_count'],
            bins=[-1, 0, 2, float('inf')],
            labels=['None', 'Low (1-2)', 'High (3+)']
        )
        
        # Calculate statistics
        high_service_flights = flight_services[flight_services['special_service_count'] >= 3]
        high_service_percentage = (len(high_service_flights) / len(flight_services)) * 100
        
        # Delay analysis by service category
        service_delays = flight_services.groupby('service_category')['departure_delay_minutes'].agg([
            'mean', 'median', 'count'
        ]).round(2)
        
        print(f" SPECIAL SERVICES STATISTICS:")
        print(f"    High special service flights (3+): {high_service_percentage:.1f}%")
        print(f"    Average special services per flight: {flight_services['special_service_count'].mean():.2f}")
        
        print(f"\n DELAY ANALYSIS BY SERVICE CATEGORY:")
        print(service_delays)
        
        # Additional analysis
        print(f"\n SPECIAL SERVICE BREAKDOWN:")
        service_types = self.pnr_remarks['special_service_request'].value_counts()
        print("Top special service requests:")
        for service, count in service_types.head(5).items():
            print(f"    {service}: {count:,} requests")
        
        return {
            'high_service_percentage': high_service_percentage,
            'avg_services_per_flight': flight_services['special_service_count'].mean(),
            'service_delay_analysis': service_delays
        }
    
    def run_complete_eda(self):
        """Run the complete EDA analysis"""
        print("UNITED AIRLINES FLIGHT DIFFICULTY SCORE - EDA")
        print("="*60)
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Run all EDA questions
        results = {}
        
        results['delay'] = self.question_1_delay_analysis()
        results['ground_time'] = self.question_2_ground_time_analysis()
        results['baggage'] = self.question_3_baggage_analysis()
        results['passenger_load'] = self.question_4_passenger_load_analysis()
        results['special_services'] = self.question_5_special_services_analysis()
        
        # Summary report
        print("\n" + "="*60)
        print("EDA SUMMARY REPORT")
        print("="*60)
        
        print(f"\n KEY FINDINGS:")
        print(f"    Average departure delay: {results['delay']['avg_delay']:.1f} minutes")
        print(f"    Late departure rate: {results['delay']['late_percentage']:.1f}%")
        print(f"    Tight ground time flights: {results['ground_time']['tight_ground_time_percentage']:.1f}%")
        print(f"    Average transfer bag ratio: {results['baggage']['avg_transfer_ratio']:.3f}")
        print(f"    Average load factor: {results['passenger_load']['avg_load_factor']:.3f}")
        print(f"    High special service flights: {results['special_services']['high_service_percentage']:.1f}%")
        
        print(f"\n EDA Analysis completed successfully!")
        
        return results

def main():
    """Main function to run the EDA analysis"""
    # Initialize EDA analyzer
    eda = FlightEDA(data_path="../data/")
    
    # Run complete analysis
    results = eda.run_complete_eda()
    
    return results

if __name__ == "__main__":
    results = main()
