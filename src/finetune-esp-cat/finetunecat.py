import os
import pandas as pd
from datasets import load_from_disk, DatasetDict
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
import torch
from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np
import wandb

os.environ["WANDB_API_KEY"] = "c4a402ca114df78f06a5cb61353abd71e704a073"

# Load the preprocessed dataset
dataset_path = "/fhome/pmlai03/preprocessed_audio_dataset"
dataset = load_from_disk(dataset_path)

# Split the dataset
dataset = dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})

# Split the dataset into train and validation
train_data = dataset["train"]
val_data = dataset["validation"]

# Load the locally saved model and processor
local_model_path = "/fhome/pmlai03/AMLALEX/FINETUNE-ESP-CA/modelesp"
model = Wav2Vec2ForCTC.from_pretrained(local_model_path)
processor = Wav2Vec2Processor.from_pretrained(local_model_path)


# Define data collator
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        # Separate inputs and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Pad inputs
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.tokenizer.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # Replace padding with -100
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels

        return batch

# Initialize the data collator
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


# Define the training arguments
training_args = TrainingArguments(
    output_dir="./fine-esp-to-cat",
    evaluation_strategy="steps",
    num_train_epochs=15,
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=100,
    warmup_steps=100,
    fp16=True,
    gradient_checkpointing=True,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=processor.feature_extractor,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-esp-to-cat")
processor.save_pretrained("./fine-esp-to-cat")



