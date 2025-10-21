#!/usr/bin/env python3
"""
Preprocess Reddit Depression Severity Dataset

This script processes the Reddit Depression Severity Dataset and creates examples
for evaluating the DPRF framework on the task of generating Reddit posts that
match the depression severity level of the original posts.
"""

import os
import pandas as pd
import argparse
import random
from tqdm import tqdm
from pathlib import Path

# Define ensure_directory locally to avoid import issues
def ensure_directory(directory_path):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)

def load_depression_data(data_path):
    """
    Load the Reddit Depression Severity Dataset.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        DataFrame with the dataset
    """
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Print dataset information
    print(f"Loaded {len(df)} examples from the depression dataset")
    print(f"Depression severity levels: {df['label'].unique().tolist()}")
    
    return df

def clean_post(text):
    """
    Clean the post text.
    
    Args:
        text: Post text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text

def create_depression_examples(depression_df, min_post_length=50, seed=42):
    """
    Create examples for the depression severity prediction task.
    
    Args:
        depression_df: DataFrame with the depression dataset
        min_post_length: Minimum length of posts to include (in characters)
        seed: Random seed for shuffling
        
    Returns:
        List of examples
    """
    examples = []
    
    # Group by depression severity level
    grouped = depression_df.groupby('label')
    
    # Print number of examples per severity level
    for level, group in grouped:
        print(f"  Level '{level}': {len(group)} examples")
    
    # Process each example
    for _, row in tqdm(depression_df.iterrows(), total=len(depression_df), desc="Processing examples"):
        text = row['text']
        label = row['label']
        
        # Clean the post
        cleaned_text = clean_post(text)
        
        # Skip posts that are too short after cleaning
        if len(cleaned_text) < min_post_length:
            continue
        
        # Create example
        example = {
            "id": str(len(examples)),
            "post": cleaned_text,
            "depression_level": label,
            "task_instruction": f"Generate a Reddit post expressing thoughts and feelings consistent with someone experiencing {label} depression.",
            # Include a simple generic persona based on the depression level
            "initial_persona": f"You are a Reddit user who experiences {label} depression. You're using the platform to express your thoughts, feelings, and experiences.",
            # Save the raw data for reference
            "raw": {
                "text": text,
                "label": label
            }
        }
        
        examples.append(example)
    
    # Shuffle the examples
    random.seed(seed)
    random.shuffle(examples)
    
    return examples

def save_examples(examples, output_path, filename="depression_data.json"):
    """
    Save examples to a JSON file.
    
    Args:
        examples: List of examples
        output_path: Path to save the examples
        filename: Name of the output file
    """
    import json
    
    output_file = os.path.join(output_path, filename)
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"Saved {len(examples)} examples to {output_file}")

def main():
    """
    Main function to preprocess the depression dataset.
    """
    parser = argparse.ArgumentParser(description='Preprocess Reddit Depression Severity Dataset')
    parser.add_argument('--raw_data_path', type=str, default='data/raw/depression_dataset.csv',
                        help='Path to the raw depression dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Directory to save processed data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--min_post_length', type=int, default=50,
                        help='Minimum length of posts to include (in characters)')
    parser.add_argument('--output_filename', type=str, default='depression_data.json',
                        help='Name of the output file containing all processed examples')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_directory(args.output_dir)
    
    # Load depression dataset
    print(f"Loading depression data from {args.raw_data_path}...")
    depression_df = load_depression_data(args.raw_data_path)
    
    # Create examples
    print("Creating examples...")
    examples = create_depression_examples(depression_df, args.min_post_length, args.seed)
    print(f"Created {len(examples)} examples")
    
    # Save examples to a single file
    print("Saving examples...")
    save_examples(examples, args.output_dir, args.output_filename)
    
    print("Done!")

if __name__ == '__main__':
    main() 