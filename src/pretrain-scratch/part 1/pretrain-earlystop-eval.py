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
import os
from datasets import concatenate_datasets

# Random seed for reproducibility
torch.manual_seed(0)

# Initialize WandB
os.environ["WANDB_API_KEY"] = "c4a402ca114df78f06a5cb61353abd71e704a073"
wandb.init(project="eval_pretraining2", config={"epochs": 15, "learning_rate": 1e-4, "batch_size": 16})

# Load and combine preprocessed datasets
chunk_dir = "/fhome/pmlai03/AMLALEX/preprocessed_audio_dataset_chunks_ca7"
chunk_files = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.startswith("chunk_")]
datasets = [load_from_disk(chunk_file) for chunk_file in chunk_files]
combined_dataset = concatenate_datasets(datasets)

dataset1 = load_from_disk("/fhome/pmlai03/AMLALEX/preprocessed_audio_dataset_ca4")
dataset2 = load_from_disk("/fhome/pmlai03/AMLALEX/preprocessed_audio_dataset_ca5")
dataset = concatenate_datasets([combined_dataset, dataset2, dataset1])

print(f"Combined Dataset: {dataset}")
print(f"Number of samples: {len(dataset)}")

# Split into train and validation sets
train_test_split = dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    "train": train_test_split["train"],
    "validation": train_test_split["test"]
})

# Wav2Vec2 configuration for pretraining
config = Wav2Vec2Config(
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    feat_proj_dropout=0.1,
    layer_norm_eps=1e-5,
    activation_dropout=0.1,
    intermediate_size=3072,
    mask_time_prob=0.065,
    mask_time_length=10,
    vocab_size=42,  # Adjusted for the refined Catalan vocabulary
)

# Initialize the Wav2Vec2 model
model = Wav2Vec2ForPreTraining(config)

# Data collator for batching
@dataclass
class DataCollatorForWav2Vec2PreTraining:
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_values = [torch.tensor(feature["audio"], dtype=torch.float32) for feature in features]
        input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
        return {"input_values": input_values}

data_collator = DataCollatorForWav2Vec2PreTraining()

# Custom Trainer with validation metrics
class Wav2Vec2PretrainingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_values = inputs["input_values"]
        outputs = model(input_values=input_values)
        
        logits = outputs.projected_states.view(-1, outputs.projected_states.size(-1))
        quantized_logits = outputs.projected_quantized_states.view(-1, outputs.projected_quantized_states.size(-1))
        
        # Contrastive loss
        contrastive_loss = -torch.sum(F.cosine_similarity(logits, quantized_logits)) / logits.size(0)
        
        # Diversity loss (optional)
        diversity_loss = outputs.codevector_perplexity.mean()
        
        loss = contrastive_loss + 0.1 * diversity_loss
        
        if return_outputs:
            return (loss, {"contrastive_loss": contrastive_loss, "diversity_loss": diversity_loss})
        return loss

    def evaluate(self, eval_dataset=None):
        eval_dataset = eval_dataset or self.eval_dataset
        dataloader = DataLoader(eval_dataset, batch_size=self.args.per_device_eval_batch_size, collate_fn=self.data_collator)
        
        total_contrastive_loss = 0
        total_diversity_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            model.eval()
            with torch.no_grad():
                outputs = model(input_values=batch["input_values"])
                logits = outputs.projected_states.view(-1, outputs.projected_states.size(-1))
                quantized_logits = outputs.projected_quantized_states.view(-1, outputs.projected_quantized_states.size(-1))
                
                contrastive_loss = -torch.sum(F.cosine_similarity(logits, quantized_logits)) / logits.size(0)
                diversity_loss = outputs.codevector_perplexity.mean()
            
            total_contrastive_loss += contrastive_loss.item()
            total_diversity_loss += diversity_loss.item()
            num_batches += 1
        
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_diversity_loss = total_diversity_loss / num_batches

        wandb.log({
            "validation_contrastive_loss": avg_contrastive_loss,
            "validation_diversity_loss": avg_diversity_loss,
        })

        return {
            "validation_contrastive_loss": avg_contrastive_loss,
            "validation_diversity_loss": avg_diversity_loss,
        }

# Training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-pretraining-checkpoints",
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    num_train_epochs=6,
    logging_dir="./logs",
    report_to="wandb",
    logging_steps=50,
    fp16=True,
    gradient_accumulation_steps=2,
    remove_unused_columns=False,
    save_total_limit=2,
    weight_decay=0.01,
    warmup_steps=50,
)

# Initialize the trainer
trainer = Wav2Vec2PretrainingTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# Training
trainer.train()

# Finish WandB session
wandb.finish()
