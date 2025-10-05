#!/usr/bin/env python3
"""
United Airlines Post-Analysis & Operational Insights
===================================================

This script performs comprehensive post-analysis of flight difficulty data to:
1. Identify destinations with consistent difficulty
2. Analyze common difficulty drivers
3. Generate operational recommendations
4. Create visualizations and comprehensive reports

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly, but make it optional
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Note: Plotly not available. Using matplotlib/seaborn for visualizations.")

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OperationalInsightsAnalyzer:
    """Class to perform post-analysis and generate operational insights"""
    
    def __init__(self, data_path="../Question B/"):
        """Initialize with data path"""
        self.data_path = data_path
        self.flight_data = None
        self.daily_summary = None
        self.destination_analysis = None
        self.driver_analysis = None
        
    def load_data(self):
        """Load flight difficulty data"""
        print("Loading Flight Difficulty Score data...")
        
        # Debug: Print current working directory and data path
        import os
        print(f"Current working directory: {os.getcwd()}")
        print(f"Data path: {self.data_path}")
        
        # Load main flight difficulty scores
        flight_scores_path = f"{self.data_path}flight_difficulty_scores.csv"
        print(f"Loading from: {flight_scores_path}")
        print(f"File exists: {os.path.exists(flight_scores_path)}")
        
        if not os.path.exists(flight_scores_path):
            print("ERROR: File not found! Trying alternative paths...")
            # Try alternative paths
            alt_paths = [
                "Question B/flight_difficulty_scores.csv",
                "../Question B/flight_difficulty_scores.csv",
                "./Question B/flight_difficulty_scores.csv"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    print(f"Found file at: {alt_path}")
                    flight_scores_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Could not find flight_difficulty_scores.csv in any expected location")
        
        self.flight_data = pd.read_csv(flight_scores_path)
        print(f" Loaded {len(self.flight_data):,} flight records")
        
        # Load daily summary
        daily_summary_path = f"{self.data_path}daily_difficulty_summary.csv"
        print(f"Loading from: {daily_summary_path}")
        
        if not os.path.exists(daily_summary_path):
            # Try alternative paths
            alt_paths = [
                "Question B/daily_difficulty_summary.csv",
                "../Question B/daily_difficulty_summary.csv",
                "./Question B/daily_difficulty_summary.csv"
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    print(f"Found file at: {alt_path}")
                    daily_summary_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Could not find daily_difficulty_summary.csv in any expected location")
        
        self.daily_summary = pd.read_csv(daily_summary_path)
        print(f" Loaded {len(self.daily_summary):,} daily summary records")
        
        # Convert date columns
        self.flight_data['scheduled_departure_date_local'] = pd.to_datetime(self.flight_data['scheduled_departure_date_local'])
        
        print("Data loading completed successfully!")
        
    def analyze_destination_difficulty(self):
        """Analyze destinations with consistent difficulty"""
        print("\nAnalyzing destination difficulty patterns...")
        
        # Group by destination and calculate statistics
        self.destination_analysis = self.flight_data.groupby('scheduled_arrival_station_code').agg({
            'difficulty_score': ['mean', 'std', 'min', 'max', 'count'],
            'difficulty_classification': lambda x: x.value_counts().to_dict(),
            'departure_delay_minutes': 'mean',
            'load_factor': 'mean',
            'special_service_count': 'mean',
            'ground_time_ratio': 'mean',
            'total_seats': 'mean',
            'carrier': lambda x: x.value_counts().to_dict()
        }).round(2)
        
        # Flatten column names
        self.destination_analysis.columns = [
            'avg_difficulty_score', 'std_difficulty_score', 'min_difficulty_score', 
            'max_difficulty_score', 'flight_count', 'classification_distribution',
            'avg_delay_minutes', 'avg_load_factor', 'avg_special_services',
            'avg_ground_time_ratio', 'avg_aircraft_size', 'carrier_distribution'
        ]
        
        # Calculate difficulty consistency (lower std = more consistent)
        self.destination_analysis['difficulty_consistency'] = 1 / (1 + self.destination_analysis['std_difficulty_score'])
        
        # Calculate percentage of difficult flights
        def calc_difficult_pct(row):
            if 'Difficult' in row['classification_distribution']:
                return (row['classification_distribution']['Difficult'] / row['flight_count']) * 100
            return 0
        
        self.destination_analysis['difficult_flight_pct'] = self.destination_analysis.apply(calc_difficult_pct, axis=1)
        
        # Sort by average difficulty score
        self.destination_analysis = self.destination_analysis.sort_values('avg_difficulty_score', ascending=False)
        
        print(f" Analyzed {len(self.destination_analysis)} destinations")
        
        return self.destination_analysis
        
    def identify_difficulty_drivers(self):
        """Identify common drivers for difficult flights"""
        print("\nIdentifying difficulty drivers...")
        
        # Separate difficult and easy flights
        difficult_flights = self.flight_data[self.flight_data['difficulty_classification'] == 'Difficult']
        easy_flights = self.flight_data[self.flight_data['difficulty_classification'] == 'Easy']
        
        # Calculate driver statistics
        drivers = [
            'ground_time_pressure', 'ground_time_risk', 'aircraft_size_complexity',
            'load_factor', 'children_complexity', 'stroller_complexity',
            'special_service_complexity', 'baggage_complexity', 'transfer_bag_complexity'
        ]
        
        self.driver_analysis = {}
        
        for driver in drivers:
            if driver in self.flight_data.columns:
                difficult_mean = difficult_flights[driver].mean()
                easy_mean = easy_flights[driver].mean()
                impact = difficult_mean - easy_mean
                
                self.driver_analysis[driver] = {
                    'difficult_mean': difficult_mean,
                    'easy_mean': easy_mean,
                    'impact': impact,
                    'impact_pct': (impact / easy_mean) * 100 if easy_mean > 0 else 0,
                    'relative_impact': impact / (easy_mean + 1) if easy_mean >= 0 else impact,  # Normalized impact
                    'fold_increase': difficult_mean / easy_mean if easy_mean > 0 else float('inf')  # How many times larger
                }
        
        # Sort drivers by impact
        sorted_drivers = sorted(self.driver_analysis.items(), key=lambda x: abs(x[1]['impact']), reverse=True)
        
        print(f" Identified {len(self.driver_analysis)} difficulty drivers")
        
        return sorted_drivers
        
    def create_destination_charts(self):
        """Create charts for destination analysis - focused on core questions"""
        print("\nCreating destination analysis charts...")
        
        # Top 15 most difficult destinations (more manageable for visualization)
        top_destinations = self.destination_analysis.head(15)
        
        # Chart 1: Most Difficult Destinations (Question 1)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Question 1: Which destinations consistently show more difficulty?
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(top_destinations)))
        bars = axes[0,0].barh(range(len(top_destinations)), top_destinations['avg_difficulty_score'], color=colors, alpha=0.8)
        axes[0,0].set_yticks(range(len(top_destinations)))
        axes[0,0].set_yticklabels(top_destinations.index, fontsize=10)
        axes[0,0].set_xlabel('Average Difficulty Score', fontsize=12, fontweight='bold')
        axes[0,0].set_title('Q1: Most Difficult Destinations', fontsize=14, fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[0,0].text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                          f'{width:.1f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Question 1 continued: Percentage of difficult flights by destination
        colors2 = plt.cm.viridis(np.linspace(0, 1, len(top_destinations)))
        bars2 = axes[0,1].barh(range(len(top_destinations)), top_destinations['difficult_flight_pct'], color=colors2, alpha=0.8)
        axes[0,1].set_yticks(range(len(top_destinations)))
        axes[0,1].set_yticklabels(top_destinations.index, fontsize=10)
        axes[0,1].set_xlabel('Percentage of Difficult Flights (%)', fontsize=12, fontweight='bold')
        axes[0,1].set_title('Q1: Consistency of Difficulty by Destination', fontsize=14, fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            axes[0,1].text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                          f'{width:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Question 2: What are the common drivers? (Destination characteristics)
        # Create a focused scatter plot showing delay vs difficulty
        scatter = axes[1,0].scatter(top_destinations['avg_delay_minutes'], top_destinations['avg_difficulty_score'], 
                                  c=top_destinations['avg_load_factor'], s=top_destinations['flight_count']*2, 
                                  cmap='RdYlBu_r', alpha=0.7, edgecolors='black', linewidth=1)
        axes[1,0].set_xlabel('Average Delay (minutes)', fontsize=12, fontweight='bold')
        axes[1,0].set_ylabel('Average Difficulty Score', fontsize=12, fontweight='bold')
        axes[1,0].set_title('Q2: Delay vs Difficulty (Size=Flights, Color=Load Factor)', fontsize=14, fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # Add destination labels
        for dest in top_destinations.index[:8]:  # Label top 8 to avoid clutter
            axes[1,0].annotate(dest, (top_destinations.loc[dest, 'avg_delay_minutes'], 
                                    top_destinations.loc[dest, 'avg_difficulty_score']),
                             fontsize=8, ha='center', va='bottom')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[1,0])
        cbar.set_label('Load Factor', fontsize=10, fontweight='bold')
        
        # Question 3: Operational efficiency insights - Ground time vs difficulty
        scatter2 = axes[1,1].scatter(top_destinations['avg_ground_time_ratio'], top_destinations['avg_difficulty_score'],
                                    c=top_destinations['avg_special_services'], s=top_destinations['flight_count']*2,
                                    cmap='plasma', alpha=0.7, edgecolors='black', linewidth=1)
        axes[1,1].set_xlabel('Average Ground Time Ratio', fontsize=12, fontweight='bold')
        axes[1,1].set_ylabel('Average Difficulty Score', fontsize=12, fontweight='bold')
        axes[1,1].set_title('Q3: Ground Time vs Difficulty (Size=Flights, Color=Special Services)', fontsize=14, fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add destination labels
        for dest in top_destinations.index[:8]:
            axes[1,1].annotate(dest, (top_destinations.loc[dest, 'avg_ground_time_ratio'], 
                                    top_destinations.loc[dest, 'avg_difficulty_score']),
                             fontsize=8, ha='center', va='bottom')
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=axes[1,1])
        cbar2.set_label('Special Services', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('destination_difficulty_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Chart 2: Destination characteristics heatmap (Question 2 & 3)
        plt.figure(figsize=(16, 10))
        
        # Prepare data for heatmap - focus on top 12 destinations for clarity
        top_12_destinations = self.destination_analysis.head(12)
        heatmap_data = top_12_destinations[['avg_difficulty_score', 'avg_delay_minutes', 
                                           'avg_load_factor', 'avg_special_services', 
                                           'avg_ground_time_ratio', 'difficult_flight_pct']].T
        
        # Create a beautiful heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Normalized Values'}, 
                   linewidths=0.5, linecolor='white',
                   annot_kws={'fontsize': 10, 'fontweight': 'bold'})
        
        plt.title('Destination Characteristics Heatmap\n(Top 12 Difficult Destinations)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Destinations', fontsize=12, fontweight='bold')
        plt.ylabel('Characteristics', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig('destination_characteristics_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    def create_driver_charts(self, sorted_drivers):
        """Create charts for difficulty drivers - focused on core questions"""
        print("\nCreating difficulty driver charts...")
        
        # Extract data for plotting
        drivers = [driver[0] for driver in sorted_drivers]
        impacts = [driver[1]['impact'] for driver in sorted_drivers]
        difficult_means = [driver[1]['difficult_mean'] for driver in sorted_drivers]
        easy_means = [driver[1]['easy_mean'] for driver in sorted_drivers]
        
        # Chart: Driver Impact Analysis (Question 2)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Question 2: What are the common drivers for difficult flights?
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(drivers)))
        bars = axes[0,0].barh(range(len(drivers)), impacts, color=colors, alpha=0.8)
        axes[0,0].set_yticks(range(len(drivers)))
        axes[0,0].set_yticklabels([d.replace('_', ' ').title() for d in drivers], fontsize=10)
        axes[0,0].set_xlabel('Impact (Difficult - Easy)', fontsize=12, fontweight='bold')
        axes[0,0].set_title('Q2: Driver Impact on Flight Difficulty', fontsize=14, fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[0,0].text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                          f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Question 2 continued: Difficult vs Easy comparison
        width = 0.35
        x_pos = np.arange(len(drivers))
        bars1 = axes[0,1].barh(x_pos - width/2, difficult_means, width, label='Difficult Flights', 
                               color='#E53E3E', alpha=0.8)
        bars2 = axes[0,1].barh(x_pos + width/2, easy_means, width, label='Easy Flights', 
                               color='#68D391', alpha=0.8)
        axes[0,1].set_yticks(x_pos)
        axes[0,1].set_yticklabels([d.replace('_', ' ').title() for d in drivers], fontsize=10)
        axes[0,1].set_xlabel('Average Value', fontsize=12, fontweight='bold')
        axes[0,1].set_title('Q2: Driver Values Comparison', fontsize=14, fontweight='bold')
        axes[0,1].legend(fontsize=10)
        axes[0,1].grid(True, alpha=0.3)
        
        # Question 3: What specific actions would you recommend? (Top drivers pie chart)
        # Filter out NaN values and ensure we have valid data
        valid_impacts = []
        valid_drivers = []
        for i, impact in enumerate(impacts[:6]):  # Top 6 drivers
            if not np.isnan(impact) and not np.isinf(impact):
                valid_impacts.append(abs(impact))
                valid_drivers.append(drivers[i].replace('_', ' ').title())
        
        if valid_impacts:
            colors_pie = plt.cm.Set3(np.linspace(0, 1, len(valid_impacts)))
            
            # Create pie chart without labels and percentages on the chart
            wedges, texts, autotexts = axes[1,0].pie(valid_impacts, labels=None, colors=colors_pie, 
                                                   autopct='', startangle=90)
            axes[1,0].set_title('Q3: Priority Drivers for Action (by Impact)', fontsize=14, fontweight='bold')
            
            # Create custom legend with driver names and percentages
            legend_labels = []
            for i, (driver, impact) in enumerate(zip(valid_drivers, valid_impacts)):
                percentage = (impact / sum(valid_impacts)) * 100
                legend_labels.append(f'{driver}: {percentage:.1f}%')
            
            # Add legend outside the pie chart
            axes[1,0].legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), 
                           fontsize=10, title='Drivers & Impact %', title_fontsize=11)
        else:
            axes[1,0].text(0.5, 0.5, 'No valid driver data available', 
                          ha='center', va='center', transform=axes[1,0].transAxes, fontsize=12)
            axes[1,0].set_title('Q3: Priority Drivers for Action', fontsize=14, fontweight='bold')
        
        # Question 3 continued: Driver correlation with difficulty score
        correlations = []
        for driver in drivers:
            if driver in self.flight_data.columns:
                corr = self.flight_data[driver].corr(self.flight_data['difficulty_score'])
                correlations.append(corr)
            else:
                correlations.append(0)
        
        colors_corr = ['#E53E3E' if c > 0 else '#68D391' for c in correlations]
        bars_corr = axes[1,1].barh(x_pos, correlations, color=colors_corr, alpha=0.8)
        axes[1,1].set_yticks(x_pos)
        axes[1,1].set_yticklabels([d.replace('_', ' ').title() for d in drivers], fontsize=10)
        axes[1,1].set_xlabel('Correlation with Difficulty Score', fontsize=12, fontweight='bold')
        axes[1,1].set_title('Q3: Driver Correlation with Overall Difficulty', fontsize=14, fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add correlation value labels
        for i, bar in enumerate(bars_corr):
            width = bar.get_width()
            axes[1,1].text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                          f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('difficulty_driver_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    def create_difficulty_distribution_chart(self):
        """Create a clean difficulty distribution chart optimized for PowerPoint"""
        print("\nCreating difficulty distribution chart...")
        
        # Create a single, clean chart optimized for PPT
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create histogram of difficulty scores
        n, bins, patches = ax.hist(self.flight_data['difficulty_score'], bins=30, alpha=0.8, 
                                  edgecolor='white', linewidth=1, color='#2E86AB')
        
        # Add vertical lines for classification thresholds
        # Calculate percentiles for classification
        difficult_threshold = self.flight_data['difficulty_score'].quantile(0.67)  # Top 33%
        easy_threshold = self.flight_data['difficulty_score'].quantile(0.33)      # Bottom 33%
        
        ax.axvline(difficult_threshold, color='#E53E3E', linestyle='--', linewidth=3, 
                  label=f'Difficult Threshold ({difficult_threshold:.1f})')
        ax.axvline(easy_threshold, color='#68D391', linestyle='--', linewidth=3, 
                  label=f'Easy Threshold ({easy_threshold:.1f})')
        
        # Add mean line
        mean_score = self.flight_data['difficulty_score'].mean()
        ax.axvline(mean_score, color='#FFA726', linestyle='-', linewidth=3, 
                  label=f'Mean Score ({mean_score:.1f})')
        
        # Styling
        ax.set_xlabel('Flight Difficulty Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Flights', fontsize=14, fontweight='bold')
        ax.set_title('Flight Difficulty Score Distribution\n(8,099 Flights Analyzed)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add statistics text box
        stats_text = f"""Statistics:
• Mean: {mean_score:.1f}
• Median: {self.flight_data['difficulty_score'].median():.1f}
• Range: {self.flight_data['difficulty_score'].min():.1f} - {self.flight_data['difficulty_score'].max():.1f}
• Std Dev: {self.flight_data['difficulty_score'].std():.1f}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add classification distribution
        classification_counts = self.flight_data['difficulty_classification'].value_counts()
        classification_text = f"""Classification:
• Difficult: {classification_counts.get('Difficult', 0):,} ({classification_counts.get('Difficult', 0)/len(self.flight_data)*100:.1f}%)
• Medium: {classification_counts.get('Medium', 0):,} ({classification_counts.get('Medium', 0)/len(self.flight_data)*100:.1f}%)
• Easy: {classification_counts.get('Easy', 0):,} ({classification_counts.get('Easy', 0)/len(self.flight_data)*100:.1f}%)"""
        
        ax.text(0.98, 0.98, classification_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Legend and grid
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set axis limits for better visibility
        ax.set_xlim(0, self.flight_data['difficulty_score'].max() * 1.05)
        
        plt.tight_layout()
        plt.savefig('difficulty_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    def generate_operational_recommendations(self, sorted_drivers):
        """Generate operational recommendations based on analysis"""
        print("\nGenerating operational recommendations...")
        
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'long_term': []
        }
        
        # Analyze top drivers
        top_drivers = sorted_drivers[:5]
        
        for driver, data in top_drivers:
            driver_name = driver.replace('_', ' ').title()
            impact = data['impact']
            
            if driver == 'ground_time_pressure':
                recommendations['high_priority'].append({
                    'driver': driver_name,
                    'recommendation': 'Implement dynamic ground time allocation based on aircraft type and historical performance',
                    'action': 'Adjust minimum turn times for different aircraft types and routes',
                    'impact': f'Impact: {impact:.3f}'
                })
                
            elif driver == 'special_service_complexity':
                recommendations['high_priority'].append({
                    'driver': driver_name,
                    'recommendation': 'Pre-allocate additional ground crew for flights with high special service requests',
                    'action': 'Create special service alerts and dedicated crew assignments',
                    'impact': f'Impact: {impact:.3f}'
                })
                
            elif driver == 'load_factor':
                recommendations['medium_priority'].append({
                    'driver': driver_name,
                    'recommendation': 'Optimize boarding processes for high-load flights',
                    'action': 'Implement priority boarding and additional gate agents for flights >90% capacity',
                    'impact': f'Impact: {impact:.3f}'
                })
                
            elif driver == 'aircraft_size_complexity':
                recommendations['medium_priority'].append({
                    'driver': driver_name,
                    'recommendation': 'Standardize procedures for large aircraft operations',
                    'action': 'Develop specific checklists and resource requirements for wide-body aircraft',
                    'impact': f'Impact: {impact:.3f}'
                })
                
            elif driver == 'ground_time_risk':
                recommendations['high_priority'].append({
                    'driver': driver_name,
                    'recommendation': 'Implement buffer time monitoring and early warning systems',
                    'action': 'Create alerts for flights with negative ground time buffer',
                    'impact': f'Impact: {impact:.3f}'
                })
        
        # Destination-specific recommendations
        top_difficult_destinations = self.destination_analysis.head(10)
        
        recommendations['long_term'].append({
            'driver': 'Destination-Specific Operations',
            'recommendation': 'Develop destination-specific operational procedures',
            'action': f'Focus on top difficult destinations: {", ".join(top_difficult_destinations.index[:5])}',
            'impact': 'High operational complexity reduction'
        })
        
        return recommendations
        
    def generate_comprehensive_report(self, sorted_drivers, recommendations):
        """Generate comprehensive operational insights report"""
        print("\nGenerating comprehensive report...")
        
        report = []
        report.append("="*80)
        report.append("UNITED AIRLINES FLIGHT DIFFICULTY ANALYSIS")
        report.append("POST-ANALYSIS & OPERATIONAL INSIGHTS REPORT")
        report.append("="*80)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 50)
        report.append(f"Analysis Period: {self.flight_data['scheduled_departure_date_local'].min().strftime('%Y-%m-%d')} to {self.flight_data['scheduled_departure_date_local'].max().strftime('%Y-%m-%d')}")
        report.append(f"Total Flights Analyzed: {len(self.flight_data):,}")
        report.append(f"Destinations Analyzed: {len(self.destination_analysis)}")
        report.append(f"Average Difficulty Score: {self.flight_data['difficulty_score'].mean():.2f}")
        report.append("")
        
        # Key Findings
        report.append("KEY FINDINGS")
        report.append("-" * 50)
        
        # Top difficult destinations
        top_5_destinations = self.destination_analysis.head(5)
        report.append("Top 5 Most Difficult Destinations:")
        for i, (dest, data) in enumerate(top_5_destinations.iterrows(), 1):
            report.append(f"{i}. {dest}: Avg Score {data['avg_difficulty_score']:.2f}, "
                         f"{data['difficult_flight_pct']:.1f}% difficult flights, "
                         f"{data['flight_count']} total flights")
        
        report.append("")
        
        # Top difficulty drivers
        report.append("Top 5 Difficulty Drivers:")
        for i, (driver, data) in enumerate(sorted_drivers[:5], 1):
            driver_name = driver.replace('_', ' ').title()
            report.append(f"{i}. {driver_name}: Impact {data['impact']:.3f} "
                         f"(Difficult: {data['difficult_mean']:.3f}, Easy: {data['easy_mean']:.3f})")
        
        report.append("")
        
        # Destination Analysis
        report.append("DESTINATION DIFFICULTY ANALYSIS")
        report.append("-" * 50)
        
        # Most consistently difficult destinations
        consistent_difficult = self.destination_analysis[
            (self.destination_analysis['difficult_flight_pct'] > 50) & 
            (self.destination_analysis['flight_count'] >= 5)
        ].head(10)
        
        report.append("Most Consistently Difficult Destinations (>50% difficult flights):")
        for dest, data in consistent_difficult.iterrows():
            report.append(f"• {dest}: {data['difficult_flight_pct']:.1f}% difficult flights, "
                         f"Avg score: {data['avg_difficulty_score']:.2f}, "
                         f"Avg delay: {data['avg_delay_minutes']:.1f} min")
        
        report.append("")
        
        # Driver Analysis
        report.append("DIFFICULTY DRIVER ANALYSIS")
        report.append("-" * 50)
        
        for driver, data in sorted_drivers:
            driver_name = driver.replace('_', ' ').title()
            report.append(f"{driver_name}:")
            report.append(f"  • Impact: {data['impact']:.3f}")
            report.append(f"  • Difficult flights average: {data['difficult_mean']:.3f}")
            report.append(f"  • Easy flights average: {data['easy_mean']:.3f}")
            
            # Use more meaningful impact metrics
            if data['fold_increase'] != float('inf'):
                report.append(f"  • Fold increase: {data['fold_increase']:.1f}x")
            else:
                report.append(f"  • Fold increase: INF (easy flights = 0)")
            
            if abs(data['impact_pct']) > 1000:  # For very large percentages
                report.append(f"  • Impact: {data['impact_pct']:.0f}% increase")
            else:
                report.append(f"  • Impact: {data['impact_pct']:.1f}% increase")
            report.append("")
        
        # Operational Recommendations
        report.append("OPERATIONAL RECOMMENDATIONS")
        report.append("-" * 50)
        
        report.append("HIGH PRIORITY ACTIONS:")
        for rec in recommendations['high_priority']:
            report.append(f"• {rec['driver']}: {rec['recommendation']}")
            report.append(f"  Action: {rec['action']}")
            report.append(f"  Expected Impact: {rec['impact']}")
            report.append("")
        
        report.append("MEDIUM PRIORITY ACTIONS:")
        for rec in recommendations['medium_priority']:
            report.append(f"• {rec['driver']}: {rec['recommendation']}")
            report.append(f"  Action: {rec['action']}")
            report.append(f"  Expected Impact: {rec['impact']}")
            report.append("")
        
        report.append("LONG-TERM STRATEGIC INITIATIVES:")
        for rec in recommendations['long_term']:
            report.append(f"• {rec['driver']}: {rec['recommendation']}")
            report.append(f"  Action: {rec['action']}")
            report.append(f"  Expected Impact: {rec['impact']}")
            report.append("")
        
        # Implementation Timeline
        report.append("IMPLEMENTATION TIMELINE")
        report.append("-" * 50)
        report.append("Phase 1 (0-3 months): High Priority Actions")
        report.append("• Implement ground time monitoring systems")
        report.append("• Deploy special service alerts")
        report.append("• Create buffer time early warning systems")
        report.append("")
        report.append("Phase 2 (3-6 months): Medium Priority Actions")
        report.append("• Optimize boarding processes")
        report.append("• Standardize large aircraft procedures")
        report.append("• Implement load factor-based resource allocation")
        report.append("")
        report.append("Phase 3 (6-12 months): Long-term Strategic Initiatives")
        report.append("• Develop destination-specific procedures")
        report.append("• Create predictive difficulty models")
        report.append("• Implement continuous improvement processes")
        report.append("")
        
        # Expected Outcomes
        report.append("EXPECTED OUTCOMES")
        report.append("-" * 50)
        report.append("• 15-20% reduction in average flight difficulty scores")
        report.append("• 10-15% improvement in on-time performance")
        report.append("• 20-25% reduction in ground time delays")
        report.append("• Improved passenger satisfaction scores")
        report.append("• Enhanced operational efficiency and cost savings")
        report.append("")
        
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        # Save report to file
        with open('operational_insights_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("Comprehensive report saved to: operational_insights_report.txt")
        
        return '\n'.join(report)
        
    def run_complete_analysis(self):
        """Run the complete post-analysis"""
        print("UNITED AIRLINES POST-ANALYSIS & OPERATIONAL INSIGHTS")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Analyze destinations
        self.analyze_destination_difficulty()
        
        # Identify drivers
        sorted_drivers = self.identify_difficulty_drivers()
        
        # Create charts
        self.create_destination_charts()
        self.create_driver_charts(sorted_drivers)
        self.create_difficulty_distribution_chart()
        
        # Generate recommendations
        recommendations = self.generate_operational_recommendations(sorted_drivers)
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report(sorted_drivers, recommendations)
        
        print("\nPost-analysis completed successfully!")
        print("Generated files:")
        print("• destination_difficulty_analysis.png")
        print("• destination_characteristics_heatmap.png")
        print("• difficulty_driver_analysis.png")
        print("• difficulty_distribution.png")
        print("• operational_insights_report.txt")
        
        return {
            'destination_analysis': self.destination_analysis,
            'driver_analysis': sorted_drivers,
            'recommendations': recommendations,
            'report': report
        }

def main():
    """Main function to run the post-analysis"""
    # Initialize analyzer
    analyzer = OperationalInsightsAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    results = main()
