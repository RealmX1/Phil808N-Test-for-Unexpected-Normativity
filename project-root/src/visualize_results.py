import json
import logging
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_directories()
        plt.style.use('seaborn')
        
    def setup_directories(self):
        """Ensure visualization output directory exists."""
        self.viz_dir = Path(self.config['processed_data_folder']) / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def load_metrics(self) -> Dict:
        """Load computed metrics from file."""
        metrics_file = Path(self.config['processed_data_folder']) / 'evaluation_metrics.json'
        with open(metrics_file, 'r') as f:
            return json.load(f)

    def create_plots(self):
        """Generate all visualization plots."""
        metrics = self.load_metrics()
        
        # Create individual plots
        self._plot_enforcement_distribution(metrics['enforcement_distribution'])
        self._plot_category_breakdown(metrics['category_metrics'])
        self._plot_semantic_shifts(metrics['semantic_shifts'])
        self._plot_bias_analysis(metrics['bias_analysis'])
        
        # Create summary report
        self._create_summary_report(metrics)
        
        logger.info("Generated all visualizations")

    def _plot_enforcement_distribution(self, distribution: Dict):
        """Create pie chart and bar plot of enforcement type distribution."""
        plt.figure(figsize=(12, 5))
        
        # Pie Chart
        plt.subplot(1, 2, 1)
        plt.pie(distribution['percentages'].values(), 
                labels=distribution['percentages'].keys(),
                autopct='%1.1f%%')
        plt.title('Distribution of Enforcement Types')
        
        # Bar Plot
        plt.subplot(1, 2, 2)
        bars = plt.bar(distribution['counts'].keys(), 
                      distribution['counts'].values())
        plt.title('Counts by Enforcement Type')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'enforcement_distribution.png')
        plt.close()

    def _plot_category_breakdown(self, category_metrics: Dict):
        """Create stacked bar chart of enforcement types by category."""
        categories = list(category_metrics.keys())
        enforcement_types = list(category_metrics[categories[0]]['enforcement_distribution'].keys())
        
        # Prepare data for stacked bar chart
        data = {
            'Category': [],
            'Enforcement Type': [],
            'Count': [],
            'Percentage': []
        }
        
        for category in categories:
            for e_type in enforcement_types:
                count = category_metrics[category]['enforcement_distribution'].get(e_type, 0)
                percentage = category_metrics[category]['enforcement_percentages'].get(e_type, 0)
                
                data['Category'].append(category)
                data['Enforcement Type'].append(e_type)
                data['Count'].append(count)
                data['Percentage'].append(percentage)
        
        df = pd.DataFrame(data)
        
        # Create stacked bar chart
        plt.figure(figsize=(12, 6))
        
        # Plot by counts
        plt.subplot(1, 2, 1)
        df_pivot = df.pivot(index='Category', columns='Enforcement Type', values='Count')
        df_pivot.plot(kind='bar', stacked=True)
        plt.title('Enforcement Types by Category (Counts)')
        plt.xticks(rotation=45)
        
        # Plot by percentages
        plt.subplot(1, 2, 2)
        df_pivot = df.pivot(index='Category', columns='Enforcement Type', values='Percentage')
        df_pivot.plot(kind='bar', stacked=True)
        plt.title('Enforcement Types by Category (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'category_breakdown.png')
        plt.close()

    def _plot_semantic_shifts(self, semantic_metrics: Dict):
        """Create visualization of semantic similarity distributions."""
        plt.figure(figsize=(12, 6))
        
        # Bar plot of average similarity by enforcement type
        plt.subplot(1, 2, 1)
        similarities = semantic_metrics['similarity_by_enforcement']
        bars = plt.bar(similarities.keys(), similarities.values())
        plt.title('Average Semantic Similarity by Enforcement Type')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Add overall average line
        plt.axhline(y=semantic_metrics['average_similarity'], 
                   color='r', linestyle='--', 
                   label=f"Overall Average: {semantic_metrics['average_similarity']:.2f}")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'semantic_shifts.png')
        plt.close()

    def _plot_bias_analysis(self, bias_metrics: Dict):
        """Create visualizations for bias analysis."""
        # Plot prompt type bias
        prompt_bias = bias_metrics['prompt_type_bias']
        df = pd.DataFrame(prompt_bias)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(df, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Enforcement Type Distribution by Prompt Type')
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'prompt_bias.png')
        plt.close()

        # If counterfactual analysis is implemented, add visualization here
        if bias_metrics['counterfactual_asymmetry']['asymmetry_detected']:
            self._plot_counterfactual_analysis(bias_metrics['counterfactual_asymmetry'])

    def _plot_counterfactual_analysis(self, asymmetry_data: Dict):
        """Create visualization for counterfactual asymmetry analysis."""
        # Placeholder for counterfactual visualization
        # Implementation depends on the structure of asymmetry data
        pass

    def _create_summary_report(self, metrics: Dict):
        """Create a summary report with key findings and all visualizations."""
        report_path = self.viz_dir / 'summary_report.html'
        
        html_content = f"""
        <html>
        <head>
            <title>Type 2 Normative Enforcement Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 20px 0; }}
                .visualization {{ margin: 30px 0; }}
                img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>Type 2 Normative Enforcement Analysis Report</h1>
            
            <div class="metric">
                <h2>Overview</h2>
                <p>Total Samples Analyzed: {metrics['total_samples']}</p>
            </div>
            
            <div class="visualization">
                <h2>Enforcement Distribution</h2>
                <img src="enforcement_distribution.png" alt="Enforcement Distribution">
            </div>
            
            <div class="visualization">
                <h2>Category Breakdown</h2>
                <img src="category_breakdown.png" alt="Category Breakdown">
            </div>
            
            <div class="visualization">
                <h2>Semantic Shifts</h2>
                <img src="semantic_shifts.png" alt="Semantic Shifts">
            </div>
            
            <div class="visualization">
                <h2>Bias Analysis</h2>
                <img src="prompt_bias.png" alt="Prompt Type Bias">
            </div>
            
            <div class="metric">
                <h2>Key Findings</h2>
                <ul>
                    <li>Average Semantic Similarity: {metrics['semantic_shifts']['average_similarity']:.2f}</li>
                    <li>Most Common Enforcement Type: {max(metrics['enforcement_distribution']['counts'].items(), key=lambda x: x[1])[0]}</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Created summary report at {report_path}")
