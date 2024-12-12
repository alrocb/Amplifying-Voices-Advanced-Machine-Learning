import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk, concatenate_datasets
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2ForPreTraining,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import DataLoader
import wandb

# Initialize WandB
os.environ["WANDB_API_KEY"] = "c4a402ca114df78f06a5cb61353abd71e704a073"
wandb.init(project="pretraining_eval", config={"epochs": 10, "learning_rate": 5e-5, "batch_size": 16})

# Random seed for reproducibility
torch.manual_seed(0)

# Load dataset
chunk_dir = "/fhome/amlai08/preprocessed_audio_dataset_chunks_ca6"
chunk_files = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.startswith("chunk_")]
datasets = [load_from_disk(chunk_file) for chunk_file in chunk_files]
combined_dataset = concatenate_datasets(datasets)

# Split the dataset into train and validation sets
train_test_split = combined_dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    "train": train_test_split["train"],
    "validation": train_test_split["test"]
})

# Load the pretrained model
model = Wav2Vec2ForPreTraining.from_pretrained("/fhome/amlai08/ALEX/KEEP_TRAINING/model-10ep/checkpoint-18380")

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a custom data collator for batching
@dataclass
class DataCollatorForWav2Vec2PreTraining:
    fp16: bool = False
    device: torch.device = torch.device("cpu")

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract input values (audio waveforms)
        input_values = [torch.tensor(feature["audio"], dtype=torch.float32) for feature in features]

        # Pad input values to the maximum length in the batch
        input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)

        # Cast to half-precision if fp16 is enabled
        if self.fp16:
            input_values = input_values.half()

        # Move to the specified device
        input_values = input_values.to(self.device)

        return {"input_values": input_values}

data_collator = DataCollatorForWav2Vec2PreTraining(fp16=True, device=device)

# Define a custom Trainer
class Wav2Vec2PretrainingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_values = inputs["input_values"]
        outputs = model(input_values=input_values)

        logits = outputs.projected_states
        quantized_logits = outputs.projected_quantized_states

        # Contrastive loss
        contrastive_loss = -torch.sum(F.cosine_similarity(
            logits.view(-1, logits.size(-1)),
            quantized_logits.view(-1, quantized_logits.size(-1))
        )) / logits.size(0)

        # Diversity loss
        diversity_loss = outputs.codevector_perplexity.mean()

        # Total loss
        loss = contrastive_loss + 0.1 * diversity_loss

        if return_outputs:
            return loss, {"contrastive_loss": contrastive_loss, "diversity_loss": diversity_loss}
        return loss

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=False,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=False,
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_dataset = eval_dataset or self.eval_dataset
        dataloader = self.get_eval_dataloader(eval_dataset)

        total_contrastive_loss = 0
        total_diversity_loss = 0
        total_mask_reconstruction_accuracy = 0
        num_batches = 0

        for batch in dataloader:
            model.eval()
            with torch.no_grad():
                batch_input = batch["input_values"].to(self.model.device)
                outputs = model(input_values=batch_input)
                logits = outputs.projected_states
                quantized_logits = outputs.projected_quantized_states

                # Contrastive loss
                contrastive_loss = -torch.sum(F.cosine_similarity(
                    logits.view(-1, logits.size(-1)),
                    quantized_logits.view(-1, quantized_logits.size(-1))
                )) / logits.size(0)

                # Diversity loss
                diversity_loss = outputs.codevector_perplexity.mean()

                # Mask reconstruction accuracy
                predicted_masked = logits.argmax(dim=-1)
                target_masked = quantized_logits.argmax(dim=-1)
                mask_reconstruction_accuracy = (predicted_masked == target_masked).float().mean()

                total_contrastive_loss += contrastive_loss.item()
                total_diversity_loss += diversity_loss.item()
                total_mask_reconstruction_accuracy += mask_reconstruction_accuracy.item()
                num_batches += 1

        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_diversity_loss = total_diversity_loss / num_batches
        avg_mask_reconstruction_accuracy = total_mask_reconstruction_accuracy / num_batches

        wandb.log({
            "validation_contrastive_loss": avg_contrastive_loss,
            "validation_diversity_loss": avg_diversity_loss,
            "mask_reconstruction_accuracy": avg_mask_reconstruction_accuracy,
        })

        return {
            "validation_contrastive_loss": avg_contrastive_loss,
            "validation_diversity_loss": avg_diversity_loss,
            "mask_reconstruction_accuracy": avg_mask_reconstruction_accuracy,
        }

    def training_step(self, model, inputs, *args, **kwargs):
        # Call the parent class's training_step method
        loss = super().training_step(model, inputs, *args, **kwargs)
        loss_value = loss.detach().cpu().item()

        # Log the training loss to WandB
        wandb.log({"train_loss": loss_value})

        return loss



# Define training arguments
training_args = TrainingArguments(
    output_dir="./model-10-eval",
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
    save_total_limit=2,
    weight_decay=0.01,
    warmup_steps=50,
    remove_unused_columns=False,
)

# Initialize the Trainer
trainer = Wav2Vec2PretrainingTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# Train the model
trainer.train()

# Finish WandB
wandb.finish()
