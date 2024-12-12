import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import wandb
from datasets import load_from_disk, DatasetDict
from torch.utils.data import DataLoader
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2ForPreTraining,
    TrainingArguments,
    Trainer,
)
import torch.nn.functional as F
from datasets import load_from_disk, concatenate_datasets
import os
from datasets import concatenate_datasets


os.environ["WANDB_API_KEY"] = "c4a402ca114df78f06a5cb61353abd71e704a073"

# Random seed for reproducibility
torch.manual_seed(0)

# Initialize WandB
wandb.init(project="pretraining-def-10ep", config={"epochs": 10, "learning_rate": 8e-5, "batch_size": 8})



# Path to the directory containing the preprocessed chunks
chunk_dir = "/fhome/pmlai03/AMLALEX/preprocessed_audio_dataset_chunks_ca7"

# List all chunk files in the directory
chunk_files = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.startswith("chunk_")]

# Load each chunk and combine them
datasets = [load_from_disk(chunk_file) for chunk_file in chunk_files]
combined_dataset = concatenate_datasets(datasets)

# Verify the combined dataset
print(f"Combined Dataset loaded: {combined_dataset}")


# Load the preprocessed dataset (audio only, no labels)
dataset1 = load_from_disk("/fhome/pmlai03/AMLALEX/preprocessed_audio_dataset_ca4")
print(f"Dataset1 loaded: {dataset1}")


# Load the preprocessed dataset (audio only, no labels)
dataset2 = load_from_disk("/fhome/pmlai03/AMLALEX/preprocessed_audio_dataset_ca5")
print(f"Dataset2 loaded: {dataset2}")


# Concatenate the datasets
dataset = concatenate_datasets([combined_dataset, dataset2, dataset1])

# Verify the result
print(f"Combined Dataset: {dataset}")
print(f"Number of samples: {len(dataset)}")



# Split the dataset into train and validation sets
train_test_split = dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    "train": train_test_split["train"],
    "validation": train_test_split["test"]
})

# Define Wav2Vec2 configuration for pretraining
config = Wav2Vec2Config(
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    feat_proj_dropout=0.1,
    layer_norm_eps=1e-5,
    activation_dropout=0.1,
    intermediate_size=3072,
    mask_time_prob=0.15,
    mask_time_length=10,
    vocab_size=33, 
)

# Initialize the Wav2Vec2 model for pretraining
model = Wav2Vec2ForPreTraining(config)

# Define a custom data collator for self-supervised training
@dataclass
class DataCollatorForWav2Vec2PreTraining:
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract input values (audio waveforms)
        input_values = [torch.tensor(feature["audio"], dtype=torch.float32) for feature in features]
        # Pad input values to the maximum length in the batch
        input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
        return {"input_values": input_values}

# Initialize the data collator
data_collator = DataCollatorForWav2Vec2PreTraining()

# Custom Trainer to compute the contrastive loss
class Wav2Vec2PretrainingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_values = inputs["input_values"]
        
        # Forward pass
        outputs = model(input_values=input_values)
        
        # Extract outputs
        logits = outputs.projected_states  # (batch_size, seq_len, hidden_dim)
        quantized_logits = outputs.projected_quantized_states  # (batch_size, seq_len, hidden_dim)

        # Reshape for contrastive loss calculation
        logits = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, hidden_dim)
        quantized_logits = quantized_logits.view(-1, quantized_logits.size(-1))  # (batch_size * seq_len, hidden_dim)
        
        # Contrastive loss (negative cosine similarity)
        positive_loss = -torch.sum(F.cosine_similarity(logits, quantized_logits)) / logits.size(0)
        
        # Optionally include diversity loss (e.g., codevector perplexity)
        diversity_loss = outputs.codevector_perplexity.mean()  # Example, if needed

        # Total loss
        loss = positive_loss + 0.1 * diversity_loss  # Adjust weights as needed
        
        return (loss, outputs) if return_outputs else loss

# Define training arguments
training_args = TrainingArguments(
    output_dir="./model-10ep",
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=8e-5,
    num_train_epochs=10,
    logging_dir="./logs",
    report_to="wandb",  # Log metrics to WandB
    logging_steps=50,   # Frequency of logging
    fp16=True,          # Use mixed precision if GPU supports it
    remove_unused_columns=False,
    gradient_accumulation_steps=2,
    save_total_limit=1,
    weight_decay=0.01,
)

# Initialize the custom trainer
trainer = Wav2Vec2PretrainingTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# Start training
trainer.train()

# Finish WandB
wandb.finish()
