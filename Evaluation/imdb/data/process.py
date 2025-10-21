#!/usr/bin/env python3  
"""  
Preprocess IMDB Movie Review Dataset  

This script processes the IMDB dataset (previously saved as a CSV) and creates   
structured examples suitable for tasks like generating reviews with a specific sentiment.  
"""  

import os  
import pandas as pd  
import argparse  
import random  
import json  
from tqdm import tqdm  

# Define ensure_directory locally to avoid import issues  
def ensure_directory(directory_path):  
    """  
    Ensure that a directory exists, creating it if necessary.  
    
    Args:  
        directory_path: Path to the directory  
    """  
    os.makedirs(directory_path, exist_ok=True)  

def load_imdb_data(data_path):  
    """  
    Load the pre-saved IMDB dataset CSV.  
    
    Args:  
        data_path: Path to the CSV file (e.g., 'data/raw/raw.csv')  
        
    Returns:  
        DataFrame with the dataset  
    """  
    # Check if the file exists  
    if not os.path.exists(data_path):  
        print(f"Error: Raw data file not found at {data_path}")  
        print("Please run the download script first to generate 'data/raw/raw.csv'")  
        exit(1) # Exit if the raw data is missing  
        
    # Load the dataset  
    df = pd.read_csv(data_path)  
    
    # Print dataset information  
    print(f"Loaded {len(df)} examples from the IMDB dataset")  
    # Ensure 'label' column exists and check unique values  
    if 'label' in df.columns:  
        print(f"Sentiment labels found: {df['label'].unique().tolist()}")  
    else:  
        print("Error: 'label' column not found in the CSV. Please ensure the download script saved it correctly.")  
        exit(1)  
    if 'text' not in df.columns:  
        print("Error: 'text' column not found in the CSV.")  
        exit(1)  

    return df  

def clean_text(text):  
    """  
    Clean the review text.  
    
    Args:  
        text: Review text to clean  
        
    Returns:  
        Cleaned text  
    """  
    if not isinstance(text, str):  
        return ""  
    
    # Remove extra whitespace (basic cleaning)  
    # Add more cleaning steps if needed (e.g., removing HTML tags, although datasets library often handles this)  
    text = " ".join(text.split())  
    
    return text  

def create_imdb_examples(imdb_df, min_review_length=30, seed=42):  
    """  
    Create structured examples for sentiment analysis/generation tasks.  
    
    Args:  
        imdb_df: DataFrame with the IMDB dataset  
        min_review_length: Minimum length of reviews to include (in characters)  
        seed: Random seed for shuffling  
        
    Returns:  
        List of examples (dictionaries)  
    """  
    examples = []  
    
    # Group by sentiment label for informational purposes  
    grouped = imdb_df.groupby('label')  
    
    # Print number of examples per sentiment label  
    print("Number of examples per sentiment:")  
    for label, group in grouped:  
        sentiment = "Positive (1)" if label == 1 else "Negative (0)"  
        print(f"  Sentiment '{sentiment}': {len(group)} examples")  
    
    skipped_count = 0  
    # Process each example  
    for _, row in tqdm(imdb_df.iterrows(), total=len(imdb_df), desc="Processing IMDB examples"):  
        text = row['text']  
        label = int(row['label']) # Ensure label is integer  
        
        # Clean the review text  
        cleaned_text = clean_text(text)  
        
        # Skip reviews that are too short after cleaning  
        if len(cleaned_text) < min_review_length:  
            skipped_count += 1  
            continue  
            
        # Determine sentiment string  
        sentiment_str = "positive" if label == 1 else "negative"  
        
        # Create example dictionary  
        example = {  
            "id": str(len(examples)), # Simple incremental ID  
            "review": cleaned_text,  
            "sentiment_label": label, # 0 for negative, 1 for positive  
            "task_instruction": f"Generate a movie review expressing a {sentiment_str} sentiment about a film.",  
            # Include a simple generic persona based on the sentiment  
            "initial_persona": f"You are someone who watched a movie and wants to write a review. You felt the movie was generally {sentiment_str}.",  
            # Save the raw data for reference  
            "raw": {  
                "text": text,  
                "label": label  
            }  
        }  
        
        examples.append(example)  
        
    if skipped_count > 0:  
        print(f"Skipped {skipped_count} examples due to being shorter than {min_review_length} characters.")  
        
    # Shuffle the examples  
    random.seed(seed)  
    random.shuffle(examples)  
    
    return examples  

def save_examples(examples, output_path, filename="imdb_processed.json"):  
    """  
    Save examples to a JSON file.  
    
    Args:  
        examples: List of examples  
        output_path: Path to save the examples  
        filename: Name of the output file  
    """  
    output_file = os.path.join(output_path, filename)  
    with open(output_file, 'w', encoding='utf-8') as f:  
        json.dump(examples, f, indent=2, ensure_ascii=False) # Use ensure_ascii=False for potentially non-ASCII text  
    
    print(f"Saved {len(examples)} processed examples to {output_file}")  

def main():  
    """  
    Main function to preprocess the IMDB dataset.  
    """  
    parser = argparse.ArgumentParser(description='Preprocess IMDB Movie Review Dataset')  
    parser.add_argument('--raw_data_path', type=str, default='raw/raw.csv',  
                        help='Path to the raw IMDB dataset CSV file (output of download script)')  
    parser.add_argument('--output_dir', type=str, default='processed',  
                        help='Directory to save processed data')  
    parser.add_argument('--seed', type=int, default=42,  
                        help='Random seed for reproducibility')  
    parser.add_argument('--min_review_length', type=int, default=30,  
                        help='Minimum length of reviews to include (in characters)')  
    parser.add_argument('--output_filename', type=str, default='imdb_processed.json',  
                        help='Name of the output JSON file containing all processed examples')  
    
    args = parser.parse_args()  
    
    # Ensure output directory exists  
    ensure_directory(args.output_dir)  
    
    # Load IMDB dataset  
    print(f"Loading IMDB data from {args.raw_data_path}...")  
    imdb_df = load_imdb_data(args.raw_data_path)  
    
    # Create examples  
    print("Creating structured examples...")  
    examples = create_imdb_examples(imdb_df, args.min_review_length, args.seed)  
    print(f"Created {len(examples)} structured examples")  
    
    # Save examples to a single file  
    print("Saving examples...")  
    save_examples(examples, args.output_dir, args.output_filename)  
    
    print("Done processing IMDB dataset!")  

if __name__ == '__main__':  
    main()  