import os
import pandas as pd
from datasets import load_from_disk, DatasetDict
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Processor, TrainingArguments, Trainer
import torch
from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np
import wandb
from evaluate import load as load_metric

# Initialize WandB
os.environ["WANDB_API_KEY"] = "c4a402ca114df78f06a5cb61353abd71e704a073"
wandb.init(project="wav2vec2-finetuning-ca-def")

# Load the preprocessed dataset
dataset1 = load_from_disk("/fhome/amlai08/preprocessed_audio_dataset")

# Load the second preprocessed dataset
dataset2 = load_from_disk("/fhome/amlai08/preprocessed_audio_dataset2")

from datasets import concatenate_datasets

# Concatenate the datasets
dataset = concatenate_datasets([dataset1, dataset2])

# Split the dataset
dataset = dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})
print(dataset)
# Load the processor
processor = Wav2Vec2Processor.from_pretrained("/fhome/amlai08/wav2vec2-catalan-tokenizer")

# Add pad token if necessary
if processor.tokenizer.pad_token is None:
    processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.pad_token)

# Save and reload the tokenizer and processor to ensure consistency
processor.tokenizer.save_pretrained("/fhome/amlai08/wav2vec2-catalan-tokenizer")
processor.save_pretrained("/fhome/amlai08/wav2vec2-catalan-processor")
processor = Wav2Vec2Processor.from_pretrained("/fhome/amlai08/wav2vec2-catalan-processor")

# Load the base model from the pretraining checkpoint
base_model = Wav2Vec2Model.from_pretrained("/fhome/amlai08/ALEX/FINETUNING/model-10ep-v3/checkpoint-18000")

# Initialize Wav2Vec2ForCTC with the base model's configuration
model = Wav2Vec2ForCTC(config=base_model.config)

# Copy the pretrained weights
model.wav2vec2.load_state_dict(base_model.state_dict())

# Update model config
model.config.vocab_size = len(processor.tokenizer)
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.ctc_loss_reduction = "mean"
model.config.ctc_zero_infinity = True
model.config.ctc_loss_blank_token_id = processor.tokenizer.pad_token_id

# Resize the model's embedding layer
model.lm_head = torch.nn.Linear(model.config.hidden_size, model.config.vocab_size)
model.lm_head.apply(model._init_weights)

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
    output_dir="./wav2vec2-finetuned-ca",
    group_by_length=True,
    per_device_train_batch_size=8,  # Adjusted for memory constraints
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=1e-4,
    warmup_steps=500,
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
trainer.save_model("./wav2vec2-finetuned-ca-def")
processor.save_pretrained("./wav2vec2-finetuned-ca-def")

# Finish WandB
wandb.finish()
