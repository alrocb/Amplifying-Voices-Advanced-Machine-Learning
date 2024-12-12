import os
import pandas as pd
from datasets import load_from_disk, DatasetDict
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
import torch
from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np
import wandb
from evaluate import load as load_metric

# Initialize WandB
os.environ["WANDB_API_KEY"] = "c4a402ca114df78f06a5cb61353abd71e704a073"
wandb.init(project="wav2vec2-finetuning-ca-p3-1")

# Load the preprocessed dataset
dataset = load_from_disk("/fhome/amlai08/preprocessed_audio_dataset2")

# Split the dataset
dataset = dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})

# Load the processor
processor = Wav2Vec2Processor.from_pretrained("/fhome/amlai08/ALEX/FINETUNING-P2/finetunep2/wav2vec2-finetuned-ca-p2")

# Load the previously trained model
model = Wav2Vec2ForCTC.from_pretrained("/fhome/amlai08/ALEX/FINETUNING-P2/finetunep2/wav2vec2-finetuned-ca/checkpoint-6130")

# Verify the model configuration
print("Model PAD token ID:", model.config.pad_token_id)
print("Model vocab size:", model.config.vocab_size)
print("Model blank token ID:", model.config.ctc_loss_blank_token_id)

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

# Initialize the WER metric
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Decode the predictions and references
    pred_str = processor.batch_decode(pred_ids)
    label_ids = pred.label_ids
    # Replace -100 with the pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-finetuned-ca-p3-1",
    group_by_length=True,
    per_device_train_batch_size=8,  # Adjusted for memory constraints
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=25,  # Adjust as needed
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=9e-5,
    warmup_steps=100,
    save_total_limit=1,
    report_to="wandb",
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Save the model and processor
trainer.save_model("./wav2vec2-finetuned-ca-p3-1")
processor.save_pretrained("./wav2vec2-finetuned-ca-p3-1")

# Finish WandB
wandb.finish()
