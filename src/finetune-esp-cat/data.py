import os
import pandas as pd
from datasets import Dataset, concatenate_datasets
import torchaudio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2FeatureExtractor

# Function to load a single dataset
def load_local_dataset(data_dir):
    path = os.path.join(data_dir, "validated.tsv")
    validated_data = pd.read_csv(path, sep='\t')

    # Keep only necessary columns, such as "path" and "sentence"
    validated_data = validated_data[["path", "sentence"]]
    validated_data["path"] = validated_data["path"].apply(lambda x: os.path.join(data_dir, "clips", x))

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(validated_data)

    # Cast "path" column to Audio using a custom audio loading function
    dataset = dataset.map(
        lambda batch: {"audio": load_audio_torchaudio(batch["path"])},
        remove_columns=["path"]
    )
    return dataset

# Custom function to load audio using torchaudio
def load_audio_torchaudio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16_000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16_000)
        waveform = resampler(waveform)
    return {"array": waveform.squeeze(0).numpy(), "sampling_rate": 16_000}

# List of dataset directories
data_dirs = [
    "/fhome/pmlai03/ca/ca",
    "/fhome/pmlai03/ca2/ca",
    "/fhome/pmlai03/ca3/ca"

]

# Load all datasets and concatenate them
datasets = [load_local_dataset(data_dir) for data_dir in data_dirs]
dataset = concatenate_datasets(datasets)

# Print the result
print(dataset)

# Initialize the tokenizer and processor
tokenizer = Wav2Vec2CTCTokenizer("/fhome/pmlai03/AMLALEX/FINETUNE-ESP-CA/modelesp/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Filter out rows with invalid sentences
def filter_invalid_sentences(batch):
    return batch["sentence"] is not None and len(batch["sentence"].strip()) > 0

# Apply the filter
dataset = dataset.filter(filter_invalid_sentences)

#print after filtering
print("Dataset after filtering:", dataset)

# Preprocess the dataset
def prepare_batch(batch):
    # Process the audio array
    batch["input_values"] = processor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"]).input_values[0]
    # Tokenize the transcription by passing it as `text`
    batch["labels"] = processor.tokenizer(text_target=batch["sentence"]).input_ids
    return batch

# Apply preprocessing while retaining 'sentence' for fine-tuning
dataset = dataset.map(prepare_batch)


# Save the dataset to disk
save_path = "./preprocessed_audio_dataset"
dataset.save_to_disk(save_path)
print(f"Preprocessed dataset saved to {save_path}")
