import pandas as pd
from datasets import Dataset, Audio, concatenate_datasets
import os

# List of dataset directories
data_dirs = [
    r"D:\Usuario\Desktop\Uni\4th year\Advanced Machine Learning\PROJECT GROUP\ca",
    r"D:\Usuario\Desktop\Uni\4th year\Advanced Machine Learning\PROJECT GROUP\ca2",
    r"D:\Usuario\Desktop\Uni\4th year\Advanced Machine Learning\PROJECT GROUP\ca3",
    r"D:\Usuario\Desktop\Uni\4th year\Advanced Machine Learning\PROJECT GROUP\ca4\ca",
    r"D:\Usuario\Desktop\Uni\4th year\Advanced Machine Learning\PROJECT GROUP\ca5\ca",
    r"D:\Usuario\Desktop\Uni\4th year\Advanced Machine Learning\PROJECT GROUP\ca6\ca",
    r"D:\Usuario\Desktop\Uni\4th year\Advanced Machine Learning\PROJECT GROUP\ca7\ca",
]

def load_local_dataset(data_dir):
    path = os.path.join(data_dir, "validated.tsv")
    validated_data = pd.read_csv(path, sep='\t')

    # Keep only necessary columns (add other columns as needed)
    # If you need accents, gender, etc., you can keep them here.
    # For now, let's say you keep "accents" if present:
    columns_to_keep = ["path", "sentence"]
    if "accents" in validated_data.columns:
        columns_to_keep.append("accents")

    validated_data = validated_data[columns_to_keep]

    # Update path to absolute paths
    validated_data["path"] = validated_data["path"].apply(lambda x: os.path.join(data_dir, "clips", x))

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(validated_data)

    # Cast "path" column to audio using Audio feature
    # This ensures that loading `dataset` later will give you an `audio` column
    dataset = dataset.cast_column("path", Audio(sampling_rate=16_000))
    return dataset

if __name__ == "__main__":
    # Load all datasets and concatenate them
    datasets = [load_local_dataset(data_dir) for data_dir in data_dirs]
    dataset = concatenate_datasets(datasets)
    
    # Save this intermediate dataset
    intermediate_path = "./catalan_intermediate_dataset"
    dataset.save_to_disk(intermediate_path)
    print(f"Intermediate dataset saved at {intermediate_path}")
    print(dataset)
