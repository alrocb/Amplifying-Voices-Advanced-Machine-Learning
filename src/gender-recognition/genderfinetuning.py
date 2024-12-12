import os
from datasets import load_from_disk, DatasetDict
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, TrainingArguments, Trainer
import torch
from dataclasses import dataclass
from typing import Dict, List, Union
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Wav2Vec2Model, Wav2Vec2ForSequenceClassification
import torch.nn as nn


# Initialize WandB
wandb.init(project="wav2vec2-gender-recognition")

# Load the preprocessed dataset
dataset_path = "./preprocessed_audio_dataset2"  # Path to your dataset
dataset = load_from_disk(dataset_path)

# Add gender labels
def add_gender_label(batch):
    # Map genders to numeric labels
    gender_map = {"male": 0, "female": 1}
    batch["label"] = gender_map[batch["gender"]]
    return batch

# Apply gender label mapping
dataset = dataset.map(add_gender_label, remove_columns=["gender"])

# Split the dataset
dataset = dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})

# Load processor
processor = Wav2Vec2Processor.from_pretrained("/fhome/amlai08/ALEX/FINETUNING/model-10ep-v3")

# Define data collator
@dataclass
class DataCollatorWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        labels = [feature["label"] for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

data_collator = DataCollatorWithPadding(processor=processor, padding=True)

# Load the base model from the pretraining checkpoint
base_model = Wav2Vec2Model.from_pretrained("/fhome/amlai08/ALEX/FINETUNING/model-10ep-v3/checkpoint-18000")

# Initialize Wav2Vec2ForSequenceClassification with the base model's configuration
model = Wav2Vec2ForSequenceClassification(config=base_model.config)

# Copy the pretrained weights
model.wav2vec2.load_state_dict(base_model.state_dict())

# Update model config for classification
model.config.num_labels = 2  # Binary classification
model.config.problem_type = "single_label_classification"

# Resize the classification head
model.classifier = nn.Linear(model.config.hidden_size, model.config.num_labels)
model.classifier.apply(model._init_weights)

# Verify the model setup
print(model)


# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Define training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-gender-recognition",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    learning_rate=1e-4,
    num_train_epochs=10,
    fp16=True,
    save_total_limit=2,
    report_to="wandb",
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model and processor
trainer.save_model("./wav2vec2-gender-recognition")
processor.save_pretrained("./wav2vec2-gender-recognition")

# Finish WandB
wandb.finish()
