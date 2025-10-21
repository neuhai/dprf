import json
import random
import os
import shutil
from pathlib import Path

# File path settings
input_dir = "/home/bsun2/DPRF2/Evaluation/interview/data/processed"
output_dir = "./ablation"
os.makedirs(output_dir, exist_ok=True)

# Random seed to ensure reproducibility
random.seed(42)

# Get all JSON files from the input directory
input_path = Path(input_dir)
json_files = list(input_path.glob("*.json"))

print(f"Found {len(json_files)} JSON files in {input_dir}")

# Check if there are at least 20 JSON files
if len(json_files) < 20:
    raise ValueError(f"Not enough JSON files: only {len(json_files)} files found, need at least 20")

# Randomly select 20 JSON files
selected_files = random.sample(json_files, 20)

# Copy selected files to output directory
copied_files = []
for file_path in selected_files:
    # Keep the original filename
    output_file = Path(output_dir) / file_path.name
    
    # Copy the file
    shutil.copy2(file_path, output_file)
    copied_files.append(file_path.name)
    
    print(f"✅ Copied: {file_path.name}")

print(f"\n🎉 Successfully copied {len(copied_files)} JSON files to {output_dir}")
print(f"📁 Selected files:")
for filename in sorted(copied_files):
    print(f"   - {filename}")