import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
import os

# Initialize the model
model = SentenceTransformer('sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja')

def combine_and_normalize(fort_data):
    features = [
        fort_data['name'],
        fort_data['title'],
        fort_data['summary'],
        ' '.join(str(v) for v in fort_data['infobox_data'].values()),
        ' '.join(fort_data['images'])  # Add image URLs to the features
    ]
    return ' | '.join(filter(lambda x: x and 'nan' not in str(x).lower(), features))

def handle_nan_and_clean(obj):
    if isinstance(obj, dict):
        return {k: handle_nan_and_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [handle_nan_and_clean(v) for v in obj]
    elif pd.isna(obj) or (isinstance(obj, float) and np.isnan(obj)):
        return "Not Specified"
    elif isinstance(obj, str):
        return ''.join(filter(lambda x: x.isprintable(), obj)).strip()
    else:
        return obj

def validate_metadata(metadata):
    return {k: (v if v is not None and v != "" else "Not Specified") for k, v in metadata.items()}

def upload_to_pinecone(fort_data_list):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = "maharashtra-forts"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # Dimension for 'sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja'
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    index = pc.Index(index_name)

    batch_size = 50
    for i in tqdm(range(0, len(fort_data_list), batch_size)):
        try:
            batch = fort_data_list[i:i+batch_size]
            
            combined_features = [combine_and_normalize(fort) for fort in batch]
            embeddings = model.encode(combined_features)
            
            ids = [str(j) for j in range(i, min(i+batch_size, len(fort_data_list)))]
            vectors = embeddings.tolist()
            
            metadata = [{k: v for k, v in fort.items() if k != 'infobox_data'} for fort in batch]
            for j, meta in enumerate(metadata):
                meta = handle_nan_and_clean(meta)
                meta = validate_metadata(meta)
                metadata[j] = meta

            to_upsert = list(zip(ids, vectors, metadata))
            index.upsert(vectors=to_upsert)
            
        except Exception as e:
            print(f"Error during upsert for batch starting at index {i}: {e}")
            print(f"Problematic metadata: {json.dumps(metadata, indent=2, default=str)}")
            continue

    print("Data embedding and storage complete!")

def main():
    # Load the JSON file
    with open("maharashtra_forts.json", "r", encoding="utf-8") as f:
        all_fort_data = json.load(f)

    if not all_fort_data:
        print("No fort data found in the JSON file. Exiting.")
        return

    print(f"Loaded data for {len(all_fort_data)} forts from maharashtra_forts.json")

    # Upload data to Pinecone
    upload_to_pinecone(all_fort_data)

if __name__ == "__main__":
    main()