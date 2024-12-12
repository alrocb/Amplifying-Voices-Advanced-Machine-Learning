import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import wandb
from datasets import load_from_disk
import torch.nn.functional as F
from datasets import load_from_disk, concatenate_datasets
import os
from datasets import concatenate_datasets

# Random seed for reproducibility
torch.manual_seed(0)

# Initialize wandb
wandb.init(project="wav2vec2_pretraining_custom", config={"epochs": 50, "learning_rate": 3e-4, "batch_size": 8})

# Check GPU availability
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()))


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


# Define PyTorch Dataset for DataLoader
class AudioDataset:
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Extract and return the audio waveform
        return torch.tensor(self.dataset[idx]["audio"], dtype=torch.float32)

# Create PyTorch dataset and DataLoader
audio_dataset = AudioDataset(dataset)
train_dataloader = DataLoader(audio_dataset, batch_size=8, collate_fn=lambda x: {
    "input_values": torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
})

# Define Wav2Vec2 configuration and model for pretraining
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
    vocab_size=32,
)
model = Wav2Vec2ForPreTraining(config)

# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=3e-4)
num_epochs = 50
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Initialize Accelerator for mixed precision and multi-GPU support
accelerator = Accelerator(mixed_precision="fp16")
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

# Confirm model is on the correct device
print("Model device:", next(model.parameters()).device)

# Training loop with wandb logging
model.train()
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in train_dataloader:
        # Prepare the input batch
        input_values = batch["input_values"].to(accelerator.device)

        # Forward pass
        outputs = model(input_values=input_values)

        # Calculate contrastive loss
        logits = outputs.projected_states
        quantized_logits = outputs.projected_quantized_states

        # Flatten for contrastive loss
        logits = logits.view(-1, logits.size(-1))
        quantized_logits = quantized_logits.view(-1, quantized_logits.size(-1))

        # Compute contrastive loss
        positive_loss = -torch.sum(torch.nn.functional.cosine_similarity(logits, quantized_logits)) / logits.size(0)
        loss = positive_loss
        epoch_loss += loss.item()

        # Backward pass
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Log metrics to wandb
    average_epoch_loss = epoch_loss / len(train_dataloader)
    wandb.log({"epoch": epoch + 1, "loss": average_epoch_loss})
    print(f"Epoch {epoch+1} completed. Average Loss: {average_epoch_loss}")

# Save the model after training
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("./wav2vec2-pretraining")

# Finish wandb logging
wandb.finish()
