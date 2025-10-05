#!/usr/bin/env python3
"""
United Airlines Flight Difficulty Score - Exploratory Data Analysis
================================================================

This script performs comprehensive EDA on flight data from Chicago O'Hare International Airport (ORD)
to address the 5 required analysis questions for the Flight Difficulty Score project.

Author: Akshit and Atharva
Date: 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    # Set plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available - visualizations will be skipped")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available - interactive visualizations will be skipped")

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
        print(f"✓ Loaded {len(self.flights):,} flight records")
        
        # Load PNR flight level data
        self.pnr_flight = pd.read_csv(f"{self.data_path}PNR+Flight+Level+Data.csv")
        print(f"✓ Loaded {len(self.pnr_flight):,} PNR flight records")
        
        # Load PNR remarks data
        self.pnr_remarks = pd.read_csv(f"{self.data_path}PNR Remark Level Data.csv")
        print(f"✓ Loaded {len(self.pnr_remarks):,} PNR remark records")
        
        # Load bag level data
        self.bags = pd.read_csv(f"{self.data_path}Bag+Level+Data.csv")
        print(f"✓ Loaded {len(self.bags):,} bag records")
        
        # Load airports data
        self.airports = pd.read_csv(f"{self.data_path}Airports Data.csv")
        print(f"✓ Loaded {len(self.airports):,} airport records")
        
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
        
        print("✓ Data preprocessing completed!")
        
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
        early_flights = self.flights[self.flights['departure_delay_minutes'] < -15]
        early_percentage = (len(early_flights) / len(self.flights)) * 100
        
        print(f"📊 DELAY STATISTICS:")
        print(f"   • Average departure delay: {avg_delay:.1f} minutes")
        print(f"   • Median departure delay: {median_delay:.1f} minutes")
        print(f"   • Flights departing late (>15 min): {late_percentage:.1f}%")
        print(f"   • Flights departing early (>15 min): {early_percentage:.1f}%")
        
        # Create visualizations
        if MATPLOTLIB_AVAILABLE:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Flight Delay Analysis', fontsize=16, fontweight='bold')
            
            # Delay distribution
            axes[0,0].hist(self.flights['departure_delay_minutes'], bins=50, alpha=0.7, color='skyblue')
            axes[0,0].axvline(avg_delay, color='red', linestyle='--', label=f'Mean: {avg_delay:.1f} min')
            axes[0,0].axvline(median_delay, color='orange', linestyle='--', label=f'Median: {median_delay:.1f} min')
            axes[0,0].set_xlabel('Departure Delay (minutes)')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].set_title('Distribution of Departure Delays')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Delay by carrier
            carrier_delays = self.flights.groupby('carrier')['departure_delay_minutes'].mean().sort_values(ascending=False)
            carrier_delays.plot(kind='bar', ax=axes[0,1], color='lightcoral')
            axes[0,1].set_title('Average Delay by Carrier')
            axes[0,1].set_ylabel('Average Delay (minutes)')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Delay by fleet type
            fleet_delays = self.flights.groupby('fleet_type')['departure_delay_minutes'].mean().sort_values(ascending=False).head(10)
            fleet_delays.plot(kind='bar', ax=axes[1,0], color='lightgreen')
            axes[1,0].set_title('Average Delay by Fleet Type (Top 10)')
            axes[1,0].set_ylabel('Average Delay (minutes)')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Delay categories
            delay_categories = ['Early (>15 min)', 'On Time (±15 min)', 'Late (>15 min)']
            delay_counts = [early_percentage, 100 - early_percentage - late_percentage, late_percentage]
            axes[1,1].pie(delay_counts, labels=delay_categories, autopct='%1.1f%%', 
                         colors=['lightgreen', 'lightblue', 'lightcoral'])
            axes[1,1].set_title('Flight Punctuality Distribution')
            
            plt.tight_layout()
            plt.savefig('delay_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("📊 Visualization skipped - matplotlib not available")
        
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
        
        print(f"📊 GROUND TIME STATISTICS:")
        print(f"   • Flights with tight ground time (<20% buffer): {tight_percentage:.1f}%")
        print(f"   • Flights below minimum turn time: {below_minimum_percentage:.1f}%")
        print(f"   • Average ground time buffer: {avg_buffer:.1f} minutes")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Ground Time Analysis', fontsize=16, fontweight='bold')
        
        # Ground time ratio distribution
        axes[0,0].hist(self.flights['ground_time_ratio'], bins=50, alpha=0.7, color='lightblue')
        axes[0,0].axvline(1.0, color='red', linestyle='--', label='Minimum Turn Time')
        axes[0,0].axvline(1.2, color='orange', linestyle='--', label='20% Buffer')
        axes[0,0].set_xlabel('Ground Time Ratio (Scheduled/Minimum)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distribution of Ground Time Ratios')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Ground time buffer distribution
        axes[0,1].hist(self.flights['ground_time_buffer'], bins=50, alpha=0.7, color='lightgreen')
        axes[0,1].axvline(0, color='red', linestyle='--', label='Zero Buffer')
        axes[0,1].set_xlabel('Ground Time Buffer (minutes)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Distribution of Ground Time Buffer')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Ground time by carrier
        carrier_ground = self.flights.groupby('carrier')['ground_time_ratio'].mean().sort_values(ascending=False)
        carrier_ground.plot(kind='bar', ax=axes[1,0], color='lightcoral')
        axes[1,0].axhline(1.2, color='red', linestyle='--', label='20% Buffer Line')
        axes[1,0].set_title('Average Ground Time Ratio by Carrier')
        axes[1,0].set_ylabel('Ground Time Ratio')
        axes[1,0].legend()
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Ground time categories
        ground_categories = ['Below Minimum', 'Tight (<20% buffer)', 'Adequate (≥20% buffer)']
        ground_counts = [
            below_minimum_percentage,
            tight_percentage - below_minimum_percentage,
            100 - tight_percentage
        ]
        axes[1,1].pie(ground_counts, labels=ground_categories, autopct='%1.1f%%',
                      colors=['lightcoral', 'orange', 'lightgreen'])
        axes[1,1].set_title('Ground Time Categories')
        
        plt.tight_layout()
        plt.savefig('ground_time_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
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
        
        # Aggregate bag data by flight
        bag_summary = self.bags.groupby([
            'company_id', 'flight_number', 'scheduled_departure_date_local'
        ]).agg({
            'bag_type': 'count',
            'bag_tag_unique_number': 'count'
        }).reset_index()
        
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
        
        print(f"📊 BAGGAGE STATISTICS:")
        print(f"   • Average transfer bag ratio: {avg_transfer_ratio:.3f}")
        print(f"   • Average checked bag ratio: {avg_checked_ratio:.3f}")
        print(f"   • Average bags per flight: {avg_bags_per_flight:.1f}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Baggage Analysis', fontsize=16, fontweight='bold')
        
        # Transfer ratio distribution
        axes[0,0].hist(flight_bags['transfer_ratio'].dropna(), bins=30, alpha=0.7, color='lightblue')
        axes[0,0].axvline(avg_transfer_ratio, color='red', linestyle='--', 
                         label=f'Mean: {avg_transfer_ratio:.3f}')
        axes[0,0].set_xlabel('Transfer Bag Ratio')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distribution of Transfer Bag Ratios')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Bags per flight distribution
        axes[0,1].hist(flight_bags['total_bags'].dropna(), bins=30, alpha=0.7, color='lightgreen')
        axes[0,1].axvline(avg_bags_per_flight, color='red', linestyle='--',
                         label=f'Mean: {avg_bags_per_flight:.1f}')
        axes[0,1].set_xlabel('Total Bags per Flight')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Distribution of Bags per Flight')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Transfer vs Checked ratio scatter
        axes[1,0].scatter(flight_bags['checked_ratio'], flight_bags['transfer_ratio'], 
                         alpha=0.6, color='purple')
        axes[1,0].set_xlabel('Checked Bag Ratio')
        axes[1,0].set_ylabel('Transfer Bag Ratio')
        axes[1,0].set_title('Transfer vs Checked Bag Ratios')
        axes[1,0].grid(True, alpha=0.3)
        
        # Bag type distribution
        bag_type_counts = self.bags['bag_type'].value_counts()
        axes[1,1].pie(bag_type_counts.values, labels=bag_type_counts.index, 
                     autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
        axes[1,1].set_title('Overall Bag Type Distribution')
        
        plt.tight_layout()
        plt.savefig('baggage_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
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
        
        print(f"📊 PASSENGER LOAD STATISTICS:")
        print(f"   • Average load factor: {avg_load_factor:.3f}")
        print(f"   • High load flights (>80%): {high_load_percentage:.1f}%")
        print(f"   • Load factor vs delay correlation: {load_delay_corr:.3f}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Passenger Load Analysis', fontsize=16, fontweight='bold')
        
        # Load factor distribution
        axes[0,0].hist(flight_passengers['load_factor'].dropna(), bins=30, alpha=0.7, color='lightblue')
        axes[0,0].axvline(avg_load_factor, color='red', linestyle='--',
                         label=f'Mean: {avg_load_factor:.3f}')
        axes[0,0].axvline(0.8, color='orange', linestyle='--', label='High Load (80%)')
        axes[0,0].set_xlabel('Load Factor')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distribution of Load Factors')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Load factor vs delay scatter
        axes[0,1].scatter(flight_passengers['load_factor'], flight_passengers['departure_delay_minutes'],
                         alpha=0.6, color='purple')
        axes[0,1].set_xlabel('Load Factor')
        axes[0,1].set_ylabel('Departure Delay (minutes)')
        axes[0,1].set_title(f'Load Factor vs Delay (Corr: {load_delay_corr:.3f})')
        axes[0,1].grid(True, alpha=0.3)
        
        # Children ratio distribution
        axes[1,0].hist(flight_passengers['children_ratio'].dropna(), bins=30, alpha=0.7, color='lightgreen')
        axes[1,0].set_xlabel('Children Ratio')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Distribution of Children Ratios')
        axes[1,0].grid(True, alpha=0.3)
        
        # Load factor by carrier
        carrier_load = flight_passengers.groupby('carrier')['load_factor'].mean().sort_values(ascending=False)
        carrier_load.plot(kind='bar', ax=axes[1,1], color='lightcoral')
        axes[1,1].set_title('Average Load Factor by Carrier')
        axes[1,1].set_ylabel('Load Factor')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('passenger_load_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
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
        
        print(f"📊 SPECIAL SERVICES STATISTICS:")
        print(f"   • High special service flights (3+): {high_service_percentage:.1f}%")
        print(f"   • Average special services per flight: {flight_services['special_service_count'].mean():.2f}")
        
        print(f"\n📊 DELAY ANALYSIS BY SERVICE CATEGORY:")
        print(service_delays)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Special Services Analysis', fontsize=16, fontweight='bold')
        
        # Special service count distribution
        axes[0,0].hist(flight_services['special_service_count'], bins=20, alpha=0.7, color='lightblue')
        axes[0,0].set_xlabel('Special Service Count per Flight')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distribution of Special Service Requests')
        axes[0,0].grid(True, alpha=0.3)
        
        # Delay by service category
        service_delays['mean'].plot(kind='bar', ax=axes[0,1], color='lightcoral')
        axes[0,1].set_title('Average Delay by Service Category')
        axes[0,1].set_ylabel('Average Delay (minutes)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Special service ratio vs delay
        axes[1,0].scatter(flight_services['special_service_ratio'], flight_services['departure_delay_minutes'],
                         alpha=0.6, color='purple')
        axes[1,0].set_xlabel('Special Service Ratio')
        axes[1,0].set_ylabel('Departure Delay (minutes)')
        axes[1,0].set_title('Special Service Ratio vs Delay')
        axes[1,0].grid(True, alpha=0.3)
        
        # Service category distribution
        service_counts = flight_services['service_category'].value_counts()
        axes[1,1].pie(service_counts.values, labels=service_counts.index, 
                     autopct='%1.1f%%', colors=['lightgreen', 'orange', 'lightcoral'])
        axes[1,1].set_title('Special Service Categories')
        
        plt.tight_layout()
        plt.savefig('special_services_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'high_service_percentage': high_service_percentage,
            'avg_services_per_flight': flight_services['special_service_count'].mean(),
            'service_delay_analysis': service_delays
        }
    
    def run_complete_eda(self):
        """Run the complete EDA analysis"""
        print("🛫 UNITED AIRLINES FLIGHT DIFFICULTY SCORE - EDA")
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
        
        print(f"\n📊 KEY FINDINGS:")
        print(f"   • Average departure delay: {results['delay']['avg_delay']:.1f} minutes")
        print(f"   • Late departure rate: {results['delay']['late_percentage']:.1f}%")
        print(f"   • Tight ground time flights: {results['ground_time']['tight_ground_time_percentage']:.1f}%")
        print(f"   • Average transfer bag ratio: {results['baggage']['avg_transfer_ratio']:.3f}")
        print(f"   • Average load factor: {results['passenger_load']['avg_load_factor']:.3f}")
        print(f"   • High special service flights: {results['special_services']['high_service_percentage']:.1f}%")
        
        print(f"\n✅ EDA Analysis completed successfully!")
        print(f"📁 Visualizations saved as PNG files in current directory")
        
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

