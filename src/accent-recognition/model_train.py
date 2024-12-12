from datasets import load_from_disk
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

@dataclass
class DataCollatorWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features):
        input_features = [{"input_values": f["path"]["array"]} for f in features]
        labels = [f["label"] for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

if __name__ == "__main__":
    dataset = load_from_disk("./accent_classification_dataset")
    print("Loaded final dataset:", dataset)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    base_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base")

    # Adjust model config for classification
    num_labels = len(dataset["train"].unique("label"))
    base_model.config.num_labels = num_labels
    base_model.config.problem_type = "single_label_classification"
    base_model.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
    # If the final pooled_output dimension is 256:
    base_model.classifier = nn.Linear(256, base_model.config.num_labels)

    base_model.classifier.apply(base_model._init_weights)

    data_collator = DataCollatorWithPadding(processor=processor)

    training_args = TrainingArguments(
        output_dir="./wav2vec2-accent-classification",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=50,
        learning_rate=1e-4,
        num_train_epochs=5,
        remove_unused_columns=False,
        fp16=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none"
    )

    # No tokenizer needed here, as we are dealing with audio
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model("./wav2vec2-accent-classification")
    processor.save_pretrained("./wav2vec2-accent-classification")
    print("Training completed!")
