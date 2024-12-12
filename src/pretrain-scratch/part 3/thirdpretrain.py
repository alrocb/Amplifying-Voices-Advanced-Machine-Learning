
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

from transformers import Wav2Vec2ForPreTraining
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

wandb.init(project="def-pretraining-step3", config={"epochs": 10, "learning_rate": 5e-5, "batch_size": 16})


# Random seed for reproducibility
torch.manual_seed(0)

# Load the pretrained model
model = Wav2Vec2ForPreTraining.from_pretrained("/fhome/pmlai03/AMLALEX/THIRDPRETRAINPART/model-10ep-v2/checkpoint-18000")

# Set environment variables for WandB
os.environ["WANDB_API_KEY"] = "c4a402ca114df78f06a5cb61353abd71e704a073"

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


chunk_dir = "/fhome/pmlai03/AMLALEX/preprocessed-ca6-audiochunks"

# List all chunk files in the directory
chunk_files = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.startswith("chunk_")]

# Load each chunk and combine them
datasets = [load_from_disk(chunk_file) for chunk_file in chunk_files]
dataset = concatenate_datasets(datasets)

# Verify the combined datasetÇ
print(f"Combined Dataset loaded: {dataset}")


# Split the dataset into train and validation sets
train_test_split = dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    "train": train_test_split["train"],
    "validation": train_test_split["test"]
})

# Define Wav2Vec2 configuration for pretraining
'''
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
'''

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
    output_dir="./model-10ep-v3",  # Use a directory with enough space
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=10,
    logging_dir="./logs",
    report_to="wandb",
    logging_steps=50,
    fp16=True,
    gradient_accumulation_steps=2,
    save_total_limit=1,  # Keep only the latest checkpoint
    weight_decay=0.01,
    warmup_steps=50,
    remove_unused_columns=False,
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
