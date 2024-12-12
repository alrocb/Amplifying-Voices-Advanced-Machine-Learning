import os
import pandas as pd
from datasets import Dataset, DatasetDict, Audio, concatenate_datasets, load_from_disk
import json
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torchaudio
import wandb
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
)



# Load the dataset from disk
loaded_dataset = load_from_disk("/fhome/amlai08/ALEX/processed_dataset")
print(f"Dataset loaded: {loaded_dataset}")


# Path to the saved vocab.json file
vocab_path = "/fhome/amlai08/ALEX/vocab.json"

# Load the vocabulary from the JSON file
with open(vocab_path, "r", encoding="utf-8") as vocab_file:
    vocab_dict = json.load(vocab_file)

print("Vocabulary loaded successfully!")

# check if CUDA is available
import torch
'''
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
'''




# Initialize the tokenizer with the vocabulary
tokenizer = Wav2Vec2CTCTokenizer("/fhome/amlai08/ALEX/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

# Save the tokenizer for future use
tokenizer.save_pretrained("./wav2vec2-catalan-tokenizer")

# Initialize the feature extractor for Wav2Vec2
feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000)

# Initialize processor with both feature extractor and tokenizer
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# Load the preprocessed dataset from disk
dataset = load_from_disk("/fhome/amlai08/ALEX/preprocessed_audio_dataset")
print(f"Dataset loaded: {loaded_dataset}")


# Set environment variables for WandB
os.environ["WANDB_API_KEY"] = "c4a402ca114df78f06a5cb61353abd71e704a073"



# Split the dataset (e.g., 90% train, 10% validation)
train_test_split = dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    "train": train_test_split["train"],
    "validation": train_test_split["test"]
})


# Initialize Wav2Vec2 config and model from scratch
config = Wav2Vec2Config(
    vocab_size=len(tokenizer),
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    intermediate_size=3072,
    hidden_dropout=0.1,
    activation_dropout=0.1,
    feat_proj_dropout=0.1,
    layer_norm_eps=1e-5,
    final_layer_norm=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    ctc_loss_reduction="mean",
)

model = Wav2Vec2ForCTC(config)



device = "cuda" if torch.cuda.is_available() else "cpu"
device


@dataclass
class CustomDataCollatorForCTC:
    processor: Any  # processor from Wav2Vec2Processor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract input values and labels
        input_values = [torch.tensor(feature["input_values"]) for feature in features]
        labels = [torch.tensor(feature["labels"]) for feature in features]

        # Pad input values to the maximum length in the batch
        input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)

        # Pad labels to the maximum length in the batch and replace pad tokens with -100
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)

        return {"input_values": input_values, "labels": labels}

# Initialize the custom collator
data_collator = CustomDataCollatorForCTC(processor=processor)



# Training arguments with WandB integration
training_args = TrainingArguments(
    output_dir="./wav2vec2-catalan-scratch",
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    num_train_epochs=20,
    logging_dir="./logs",
    report_to="wandb",  # Log metrics to WandB
    logging_steps=10,   # Frequency of logging
    fp16=True,          # Use if you have a compatible GPU
)

# Trainer setup
trainer = Trainer(
    model=model,
    data_collator=data_collator,  # Use DataCollatorForCTC
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.tokenizer,  # Use processor.tokenizer, not feature_extractor
)

# Start training
trainer.train()




