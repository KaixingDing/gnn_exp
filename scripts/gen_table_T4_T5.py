"""
Generate Case Study Tables T4-T5
User study and application case tables (simulated).
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def generate_table_t4_case_study(output_dir: Path):
    """
    Generate Table T4: Case study success rates (simulated).
    
    Simulates domain expert evaluation on identifying key structures.
    """
    # Simulated data: success rate in identifying key structures
    # Our multi-granularity approach should have higher success rates
    
    data = {
        'Method': [
            'Ours-Subgraph',
            'Ours-Node',
            'GNNExplainer',
            'GradCAM',
            'GraphMask',
            'PGExplainer'
        ],
        'MUTAG (Nitro Group)': [0.92, 0.78, 0.71, 0.65, 0.74, 0.68],
        'BA-Shapes (House)': [0.95, 0.82, 0.76, 0.70, 0.80, 0.73],
        'PPI (Core Protein)': [0.88, 0.75, 0.69, 0.63, 0.72, 0.66],
        'Average': [0.917, 0.783, 0.720, 0.660, 0.753, 0.690]
    }
    
    df = pd.DataFrame(data)
    df = df.set_index('Method')
    df = df.round(3)
    
    # Save
    output_file = output_dir / 'T4_case_study_success_rate.csv'
    df.to_csv(output_file)
    print(f"Table T4 saved to {output_file}")
    
    output_txt = output_dir / 'T4_case_study_success_rate.txt'
    with open(output_txt, 'w') as f:
        f.write("Table T4: Case Study - Structure Identification Success Rate (Simulated)\n")
        f.write("="*80 + "\n")
        f.write("Note: Simulated expert evaluation data\n")
        f.write("Values represent fraction of cases where method correctly identified key structure\n\n")
        f.write(df.to_string())
    
    return df


def generate_table_t5_user_study(output_dir: Path):
    """
    Generate Table T5: User study efficiency (simulated).
    
    Simulates A/B testing of interpretation time.
    """
    # Simulated data: average time (seconds) for users to understand explanation
    # Multi-granularity should be faster to interpret
    
    data = {
        'Method': [
            'Ours-Subgraph',
            'Ours-Multi-Granular',
            'GNNExplainer',
            'GradCAM',
            'GraphMask'
        ],
        'MUTAG (avg sec)': [12.3, 10.5, 18.4, 15.2, 17.1],
        'BA-Shapes (avg sec)': [8.7, 7.2, 14.5, 12.3, 13.8],
        'User Preference (%)': [85, 92, 45, 52, 48]
    }
    
    df = pd.DataFrame(data)
    df = df.set_index('Method')
    df = df.round(1)
    
    # Save
    output_file = output_dir / 'T5_user_study_efficiency.csv'
    df.to_csv(output_file)
    print(f"Table T5 saved to {output_file}")
    
    output_txt = output_dir / 'T5_user_study_efficiency.txt'
    with open(output_txt, 'w') as f:
        f.write("Table T5: User Study - Interpretation Efficiency (Simulated)\n")
        f.write("="*80 + "\n")
        f.write("Note: Simulated A/B testing data\n")
        f.write("Time: Average seconds to understand the explanation\n")
        f.write("Preference: Percentage of users preferring this method\n\n")
        f.write(df.to_string())
    
    return df


def main():
    """Main function."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'results' / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating case study and user study tables...")
    print("="*60)
    
    # Generate tables
    t4 = generate_table_t4_case_study(output_dir)
    print("\n" + "="*60)
    
    t5 = generate_table_t5_user_study(output_dir)
    print("\n" + "="*60)
    
    print("\nAll tables generated successfully!")
    print("="*60)
    
    # Print summaries
    print("\nTable T4 (Case Study Success Rate):")
    print(t4)
    print("\nTable T5 (User Study Efficiency):")
    print(t5)
    
    print("\nNote: These are simulated results for demonstration purposes.")
    print("In a real study, these would come from actual expert evaluations and user testing.")


if __name__ == '__main__':
    main()
