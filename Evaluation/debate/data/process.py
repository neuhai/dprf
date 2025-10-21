import json

def clean_bio_data(file_path):
    """
    Remove data where the bio field strictly equals 'nan is .' from JSON file
    """
    # Read JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    
    # Filter out data where bio field strictly equals 'nan is .'
    cleaned_data = []
    for item in data:
        bio = item.get('bio')
        
        # Strictly match 'nan is .'
        if bio != "nan is .":
            cleaned_data.append(item)
    
    # Calculate number of deleted entries
    deleted_count = original_count - len(cleaned_data)
    
    # Write cleaned data back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"Original data count: {original_count}")
    print(f"Deleted count: {deleted_count}")
    print(f"Remaining data count: {len(cleaned_data)}")
    print(f"Removed data where bio field equals 'nan is .'")
    print(f"File updated: {file_path}")

if __name__ == "__main__":
    file_path = "/work/hdd/bdcj/bsun2/DPRF2/Evaluation/debate/data/processed/debate_examples.json"
    clean_bio_data(file_path)