import pandas as pd
from datasets import load_from_disk, DatasetDict
from sklearn.model_selection import train_test_split

# Adjust these based on your needs
target_accents = [
    "central",
    "valencià,La Vall d'Albaida",
    "nord-occidental,Tortosí",
    "balear",
    "septentrional"
]
sample_limit = 1540
septentrional_limit = 744

def balance_class(group):
    limit = septentrional_limit if group.name == "septentrional" else sample_limit
    return group.sample(n=min(len(group), limit), random_state=42)

if __name__ == "__main__":
    intermediate_path = "./catalan_intermediate_dataset"
    dataset = load_from_disk(intermediate_path)
    print("Loaded intermediate dataset:", dataset)

    # Convert to pandas for filtering and balancing
    df = dataset.to_pandas()

    # Filter to target accents (if "accents" column available)
    df = df[df["accents"].isin(target_accents)]

    # Balance dataset
    balanced_df = df.groupby("accents").apply(balance_class).reset_index(drop=True)

    # Map accents to labels
    accent_label_map = {acc: i for i, acc in enumerate(balanced_df["accents"].unique())}
    balanced_df["label"] = balanced_df["accents"].map(accent_label_map)

    # Split into train and validation
    train_df, val_df = train_test_split(
        balanced_df,
        test_size=0.1,
        stratify=balanced_df["label"],
        random_state=42
    )

    from datasets import Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Since we casted column in step 1, the audio information should still be retrievable.
    # However, now that we recreated Datasets from Pandas, we need to recast the audio column.
    # The "path" column still points to audio files, so we can do:
    from datasets import Audio

    train_dataset = train_dataset.cast_column("path", Audio(sampling_rate=16_000))
    val_dataset = val_dataset.cast_column("path", Audio(sampling_rate=16_000))

    dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})
    dataset_dict_path = "./accent_classification_dataset"
    dataset_dict.save_to_disk(dataset_dict_path)
    print(f"Final dataset saved to {dataset_dict_path}")
    print("Accent label map:", accent_label_map)
    print(dataset_dict)
