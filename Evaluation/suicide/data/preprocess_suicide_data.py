#!/usr/bin/env python3  
"""  
Preprocessing script for the Reddit Suicide Severity dataset.  

This script processes the raw CSV dataset containing Reddit posts with suicide  
risk level annotations and converts it to a JSON format suitable for the  
evaluation framework.  
"""  

import os  
import json  
import argparse  
import pandas as pd  
import random  
from typing import Dict, List, Any, Optional  


def parse_arguments() -> argparse.Namespace:  
    """Parse command line arguments."""  
    parser = argparse.ArgumentParser(  
        description="Preprocess Reddit Suicide Severity dataset"  
    )  
    parser.add_argument(  
        "--raw_data_path",  
        type=str,  
        required=True,  
        help="Path to the raw data CSV file",  
    )  
    parser.add_argument(  
        "--output_dir",  
        type=str,  
        required=True,  
        help="Directory to save the processed data",  
    )  
    parser.add_argument(  
        "--output_filename",  
        type=str,  
        default="suicide_data.json",  
        help="Filename for the processed data JSON",  
    )  
    parser.add_argument(  
        "--seed", type=int, default=42, help="Random seed for reproducibility"  
    )  
    return parser.parse_args()  


def load_raw_data(file_path: str) -> pd.DataFrame:  
    """  
    Load the raw dataset from a CSV file.  
    
    Args:  
        file_path: Path to the CSV file containing Reddit posts and labels.  
        
    Returns:  
        DataFrame containing the loaded data.  
    """  
    try:  
        df = pd.read_csv(file_path)  
        print(f"Successfully loaded data from {file_path}")  
        print(f"Dataset shape: {df.shape}")  
        return df  
    except Exception as e:  
        print(f"Error loading data: {e}")  
        raise  


def preprocess_data(df: pd.DataFrame) -> List[Dict[str, Any]]:  
    """  
    Preprocess the raw data into the required format.  
    
    Args:  
        df: DataFrame containing the raw data.  
        
    Returns:  
        List of dictionaries containing the processed examples.  
    """  
    processed_data = []  
    
    # Check if the expected columns exist  
    if "User" in df.columns and "Post" in df.columns and "Label" in df.columns:  
        print("Using 'User' for user ID, 'Post' for content, and 'Label' for suicide risk categories")  
    else:  
        # Try to identify columns if not using expected names  
        columns = df.columns.tolist()  
        print(f"Available columns: {columns}")  
        raise ValueError("Required columns 'User', 'Post', and 'Label' not found in the dataset")  
    
    # Map suicide risk categories to standardized risk levels  
    risk_level_mapping = {  
        "Ideation": "high_risk",  
        "Attempt": "severe_risk",  
        "Behavior": "moderate_risk",  
        "Indicator": "low_risk",  
        "Supportive": "no_risk"  
    }  
    
    # Process each row in the dataframe  
    for idx, row in df.iterrows():  
        user_id = row["User"]  
        post_content = row["Post"]  
        risk_category = row["Label"]  
        
        # Skip rows with missing content  
        if pd.isna(post_content) or pd.isna(risk_category):  
            continue  
        
        # Convert risk category to standardized risk level  
        risk_level = risk_level_mapping.get(risk_category)  
        if risk_level is None:  
            print(f"Warning: Unknown risk category '{risk_category}' at index {idx}, skipping")  
            continue  
        
        # Create task instruction based on risk level  
        task_instruction = f"Generate a Reddit post expressing thoughts and feelings consistent with someone at {risk_level} of suicide."  
        
        # Create initial persona based on risk level  
        initial_persona = f"You are a Reddit user who displays characteristics consistent with {risk_level} of suicide. You're using the platform to express your thoughts, feelings, and experiences."  
        
        # Create example entry  
        example = {  
            "id": f"{idx}",  
            "post": post_content.strip(),  
            "suicide_level": risk_level,  
            "task_instruction": task_instruction,  
            "initial_persona": initial_persona,  
            "raw": {  
                "text": post_content.strip(),  
                "user": str(user_id),  
                "label": risk_category  
            }  
        }  
        
        processed_data.append(example)  
    
    print(f"Processed {len(processed_data)} valid examples")  
    return processed_data  


def save_processed_data(  
    data: List[Dict[str, Any]], output_dir: str, output_filename: str  
) -> None:  
    """  
    Save the processed data to a JSON file.  
    
    Args:  
        data: List of processed examples.  
        output_dir: Directory to save the output file.  
        output_filename: Name of the output file.  
    """  
    os.makedirs(output_dir, exist_ok=True)  
    output_path = os.path.join(output_dir, output_filename)  
    
    with open(output_path, "w") as f:  
        json.dump({"examples": data}, f, indent=2)  
    
    print(f"Saved processed data to {output_path}")  


def analyze_data(examples: List[Dict[str, Any]]) -> None:  
    """  
    Analyze the processed data and print statistics.  
    
    Args:  
        examples: List of processed examples.  
    """  
    # Count examples by risk level  
    risk_levels = {}  
    post_lengths = []  
    
    for example in examples:  
        risk_level = example["suicide_level"]  
        risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1  
        post_lengths.append(len(example["post"].split()))  
    
    print("\nData Analysis:")  
    print(f"Total examples: {len(examples)}")  
    print("Distribution by suicide risk level:")  
    for level, count in sorted(risk_levels.items()):  
        print(f"  {level}: {count} examples ({count / len(examples) * 100:.1f}%)")  
    
    if post_lengths:  
        print("\nPost length statistics (word count):")  
        print(f"  Average: {sum(post_lengths) / len(post_lengths):.1f} words")  
        print(f"  Minimum: {min(post_lengths)} words")  
        print(f"  Maximum: {max(post_lengths)} words")  


def main() -> None:  
    """Main execution function."""  
    args = parse_arguments()  
    
    # Set random seed for reproducibility  
    random.seed(args.seed)  
    
    # Load raw data  
    df = load_raw_data(args.raw_data_path)  
    
    # Preprocess data  
    processed_data = preprocess_data(df)  
    
    # Analyze processed data  
    analyze_data(processed_data)  
    
    # Save processed data  
    save_processed_data(processed_data, args.output_dir, args.output_filename)  


if __name__ == "__main__":  
    main()  