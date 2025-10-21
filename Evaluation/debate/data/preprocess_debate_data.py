#!/usr/bin/env python3
"""
Preprocess Intelligence Squared Debate Data

This script processes the Intelligence Squared (IQ2) debate transcripts
and creates examples for evaluating the DPRF framework on debate tasks.
For each speaker in a debate, it consolidates all their statements into
a comprehensive set that will be used as ground truth.
"""

import os
import re
import json
import pandas as pd
import argparse
import sys
from pathlib import Path
import random
from tqdm import tqdm

# Define ensure_directory locally to avoid import issues
def ensure_directory(directory_path):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)

def load_iq2_data(data_path):
    """
    Load Intelligence Squared data from CSV files.
    
    Args:
        data_path: Path to the directory containing the CSV files
        
    Returns:
        Tuple of DataFrames: (debates, speakers, utterances)
    """
    # Load main dataframes
    debates_df = pd.read_csv(os.path.join(data_path, 'debates.csv'))
    speakers_df = pd.read_csv(os.path.join(data_path, 'speakers.csv'))
    utterances_df = pd.read_csv(os.path.join(data_path, 'utterances.csv'))
    
    # Fix column names if needed
    if 'id' in debates_df.columns and 'debate_id' not in debates_df.columns:
        debates_df = debates_df.rename(columns={'id': 'debate_id'})
    
    if 'id' in speakers_df.columns and 'speaker_id' not in speakers_df.columns:
        speakers_df = speakers_df.rename(columns={'id': 'speaker_id'})
    
    # Set index for speakers_df
    if 'speaker_id' in speakers_df.columns:
        speakers_df = speakers_df.set_index('speaker_id')
    
    # Check if the data format is compatible
    if 'debate_id' not in utterances_df.columns or 'speaker_id' not in utterances_df.columns:
        print("ERROR: utterances.csv is missing required columns")
        print(f"Available columns: {utterances_df.columns.tolist()}")
        return None, None, None
    
    print(f"Loaded {len(debates_df)} debates, {len(speakers_df)} speakers, {len(utterances_df)} utterances")
    
    return debates_df, speakers_df, utterances_df

def clean_utterance(text):
    """
    Clean the utterance text.
    
    Args:
        text: Utterance text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove (inaudible), [laughter], etc.
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^]]*\]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_debate_topic(debate_title):
    """
    Extract the debate topic/motion from the debate title.
    
    Args:
        debate_title: Title of the debate
        
    Returns:
        Debate topic
    """
    # Common patterns in IQ2 debate titles
    if ":" in debate_title:
        return debate_title.split(":", 1)[1].strip()
    if "-" in debate_title:
        return debate_title.split("-", 1)[1].strip()
    
    # If no clear separator, just return the title
    return debate_title.strip()


def determine_speaker_position(speaker_type, debate_utterances=None, speaker_id=None):
    """
    Determine the speaker's position on the debate topic.
    
    Args:
        speaker_type: Type of speaker (e.g., "For the Motion", "Against the Motion", "Moderator")
        debate_utterances: Not used in this version, kept for compatibility
        speaker_id: Not used in this version, kept for compatibility
        
    Returns:
        Speaker's position (For, Against, Neutral/Moderator)
    """
    # If speaker_type is None or empty, default to Neutral
    if not speaker_type:
        print(f"  Speaker has no type, defaulting to Neutral")
        return "Neutral"
    
    # Print the original speaker_type for debugging
    print(f"  Speaker type: '{speaker_type}'")
    
    # Handle direct 'for' and 'against' tags from the ConvoKit data
    if speaker_type.lower() == 'for':
        return "For"
    if speaker_type.lower() == 'against':
        return "Against"
    
    # Extract position from speaker_type
    speaker_type_lower = speaker_type.lower()
    
    # More comprehensive checks for "For" position
    if ("for" in speaker_type_lower and "motion" in speaker_type_lower) or \
       ("pro" in speaker_type_lower and "motion" in speaker_type_lower) or \
       "for the motion" in speaker_type_lower or \
       "pro the motion" in speaker_type_lower or \
       "supporting" in speaker_type_lower or \
       "proponent" in speaker_type_lower or \
       "favor" in speaker_type_lower:
        return "For"
    
    # More comprehensive checks for "Against" position
    elif ("against" in speaker_type_lower and "motion" in speaker_type_lower) or \
         ("con" in speaker_type_lower and "motion" in speaker_type_lower) or \
         "against the motion" in speaker_type_lower or \
         "con the motion" in speaker_type_lower or \
         "opposing" in speaker_type_lower or \
         "opponent" in speaker_type_lower or \
         "anti" in speaker_type_lower or \
         "oppose" in speaker_type_lower:
        return "Against"
    
    # Check for moderator/host
    elif speaker_type_lower == 'mod' or \
         "moderator" in speaker_type_lower or \
         speaker_type_lower == 'host' or \
         "host" in speaker_type_lower or \
         "chairman" in speaker_type_lower or \
         "announcer" in speaker_type_lower or \
         "presenter" in speaker_type_lower:
        return "Moderator"
    
    # If there was a speaker_type but didn't match any known patterns, print it
    print(f"  Unrecognized speaker_type: '{speaker_type}', defaulting to Neutral")
    
    # Default case
    return "Neutral"

def infer_speaker_position_from_text(debate_utterances, speaker_id, debate_title):
    """
    Analyze the debate transcript to infer a speaker's position (For/Against).
    
    Args:
        debate_utterances: DataFrame with all utterances for this debate
        speaker_id: ID of the speaker to analyze
        debate_title: Title of the debate
        
    Returns:
        Inferred position (For, Against, or Neutral)
    """
    # Get utterances by this speaker
    speaker_utterances = debate_utterances[debate_utterances['speaker_id'] == speaker_id]
    
    # Look for introductory statements that typically indicate position
    intro_indicators = {}
    
    # Look at all utterances to find position indicators
    for _, utterance in debate_utterances.iterrows():
        text = utterance['text'].lower() if not pd.isna(utterance['text']) else ""
        
        # Look for phrases like "for the motion" or "against the motion"
        if speaker_id in str(text):
            position_text = None
            
            # Check for "for the motion" near speaker name
            if "for the motion" in text or "supporting the motion" in text:
                position_text = "For"
            # Check for "against the motion" near speaker name
            elif "against the motion" in text or "opposing the motion" in text:
                position_text = "Against"
                
            if position_text and speaker_id not in intro_indicators:
                intro_indicators[speaker_id] = position_text
    
    # If we found a clear indicator, return it
    if speaker_id in intro_indicators:
        return intro_indicators[speaker_id]
    
    # Look at the speaker's own utterances for clues
    position_indicators = {
        "For": 0,
        "Against": 0
    }
    
    for _, utterance in speaker_utterances.iterrows():
        text = utterance['text'].lower() if not pd.isna(utterance['text']) else ""
        
        # Look for phrases indicating agreement with the motion
        if "i support" in text or "i agree" in text or "i am for" in text:
            position_indicators["For"] += 1
            
        # Look for phrases indicating disagreement with the motion
        if "i oppose" in text or "i disagree" in text or "i am against" in text:
            position_indicators["Against"] += 1
    
    # If we have a clear indicator from their own statements
    if position_indicators["For"] > position_indicators["Against"]:
        return "For"
    elif position_indicators["Against"] > position_indicators["For"]:
        return "Against"
    
    # If we can't determine, return Neutral
    return "Neutral"

def create_consolidated_debate_examples(debates_df, speakers_df, utterances_df, min_utterances=5):
    """
    Create debate examples with consolidated statements for each speaker.
    Only includes speakers with "For" or "Against" positions, filtering out
    moderators, hosts, and neutral speakers.
    
    Args:
        debates_df: DataFrame with debate information
        speakers_df: DataFrame with speaker information
        utterances_df: DataFrame with utterance information
        min_utterances: Minimum number of utterances required for a speaker
        
    Returns:
        List of examples
    """
    examples = []
    skipped_reasons = {
        "too_few_utterances": 0,
        "too_few_clean_utterances": 0,
        "speaker_position_not_for_against": 0
    }
    speaker_positions = {}
    
    print("Creating debate examples...")
    
    # Process each debate
    for _, debate in debates_df.iterrows():
        # Get debate ID
        debate_id = debate['debate_id'] if 'debate_id' in debate else debate.name
        
        # Get debate topic from title
        debate_topic = extract_debate_topic(debate['title'])
        
        # Get utterances for this debate
        debate_utterances = utterances_df[utterances_df['debate_id'] == debate_id]
        
        if len(debate_utterances) == 0:
            print(f"  No utterances found for debate ID: {debate_id}")
            continue
            
        # Get unique speakers in this debate
        debate_speakers = debate_utterances['speaker_id'].unique()
        
        print(f"  Processing debate ID: {debate_id}, '{debate['title']}' with {len(debate_speakers)} speakers")
        
        # Process each speaker
        for speaker_id in debate_speakers:
            # Skip if speaker_id is NaN
            if pd.isna(speaker_id):
                continue
                
            # Check if speaker_id is in speakers_df index
            if speaker_id not in speakers_df.index:
                # Try to find by name if it's a string speaker_id
                if isinstance(speaker_id, str) and 'name' in speakers_df.columns:
                    matching_speakers = speakers_df[speakers_df['name'] == speaker_id]
                    if len(matching_speakers) > 0:
                        speaker = matching_speakers.iloc[0]
                    else:
                        # Create a minimal speaker record
                        speaker = pd.Series({'name': speaker_id, 'bio': '', 'occupation': ''})
                else:
                    # Create a minimal speaker record
                    speaker = pd.Series({'name': str(speaker_id), 'bio': '', 'occupation': ''})
            else:
                # Get speaker information
                speaker = speakers_df.loc[speaker_id]
            
            speaker_name = speaker['name'] if 'name' in speaker else str(speaker_id)
            
            # Get utterances by this speaker
            speaker_utterances = debate_utterances[debate_utterances['speaker_id'] == speaker_id]
            
            # Skip if too few utterances
            if len(speaker_utterances) < min_utterances:
                skipped_reasons["too_few_utterances"] += 1
                continue
            
            # Clean utterances
            cleaned_utterances = [clean_utterance(utt) for utt in speaker_utterances['text']]
            cleaned_utterances = [utt for utt in cleaned_utterances if len(utt) > 20]  # Filter out very short utterances
            
            # Skip if too few clean utterances
            if len(cleaned_utterances) < min_utterances:
                skipped_reasons["too_few_clean_utterances"] += 1
                continue
            
            # Get speaker bio
            bio = speaker['bio'] if 'bio' in speaker and not pd.isna(speaker['bio']) else ''
            if not isinstance(bio, str) or len(bio.strip()) < 10:
                # Use name and occupation if no proper bio
                occupation = speaker['occupation'] if 'occupation' in speaker and not pd.isna(speaker['occupation']) else ''
                bio = f"{speaker['name']} is {occupation}."
            
            # Get speaker type from the dataset if available
            speaker_type = None
            if 'speaker_type' in speaker_utterances.columns and not speaker_utterances['speaker_type'].isna().all():
                # Get the most common speaker_type for this speaker in this debate
                speaker_type = speaker_utterances['speaker_type'].mode()[0]
            elif 'role' in speaker.index and isinstance(speaker['role'], str) and len(speaker['role']) > 0:
                # If speaker has a role defined, use that
                speaker_type = speaker['role']
            elif 'occupation' in speaker and isinstance(speaker['occupation'], str) and 'moderator' in speaker['occupation'].lower():
                # Default to moderator based on occupation if needed
                speaker_type = "Moderator"
            
            # Determine speaker's position
            position = determine_speaker_position(speaker_type)
            
            # If position is Neutral, try to infer from the debate text
            if position == "Neutral":
                position = infer_speaker_position_from_text(debate_utterances, speaker_id, debate['title'])
                print(f"    Inferred position for {speaker_name}: {position}")
                
            # Count the positions
            if position not in speaker_positions:
                speaker_positions[position] = 0
            speaker_positions[position] += 1
            
            # Skip moderators, hosts, and neutral speakers - only keep For and Against positions
            if position not in ["For", "Against"]:
                skipped_reasons["speaker_position_not_for_against"] += 1
                continue
                
            print(f"    Adding speaker: {speaker_name} ({position})")
            
            # Consolidate all utterances into one comprehensive statement
            all_statements = "\n\n".join(cleaned_utterances)
            
            # Create an example
            example = {
                "debate_id": int(debate_id) if isinstance(debate_id, (int, float)) else debate_id,
                "debate_title": debate['title'],
                "debate_topic": debate_topic,
                "speaker_id": int(speaker_id) if isinstance(speaker_id, (int, float)) else speaker_id,
                "speaker_name": speaker['name'],
                "speaker_type": speaker_type,
                "speaker_position": position,
                "bio": bio,
                "individual_utterances": cleaned_utterances,
                "consolidated_statements": all_statements
            }
            
            examples.append(example)
    
    print(f"Speaker positions found: {speaker_positions}")
    print(f"Skipped reasons: {skipped_reasons}")
    
    return examples

def main():
    parser = argparse.ArgumentParser(description="Preprocess IQ2 debate data for DPRF evaluation")
    parser.add_argument("--raw_data_dir", default="Evaluation/debate/data/raw", help="Directory with raw IQ2 CSV files")
    parser.add_argument("--output_dir", default="Evaluation/debate/data/processed", help="Directory to save processed data")
    parser.add_argument("--min_utterances", type=int, default=5, help="Minimum utterances required for a speaker")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    ensure_directory(args.output_dir)
    
    # Load data
    debates_df, speakers_df, utterances_df = load_iq2_data(args.raw_data_dir)
    
    # Create consolidated debate examples
    examples = create_consolidated_debate_examples(
        debates_df, speakers_df, utterances_df, 
        min_utterances=args.min_utterances
    )
    
    print(f"Created {len(examples)} examples")
    
    # Save examples
    with open(os.path.join(args.output_dir, "debate_examples.json"), "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"Saved examples to {os.path.join(args.output_dir, 'debate_examples.json')}")

if __name__ == "__main__":
    main() 