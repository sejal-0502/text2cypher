# train.py
import os
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

from data import get_tokenized_datasets, MODEL_NAME

def train(output_dir, num_epochs, batch_size, learning_rate):
    
    # 1. Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # 2. Load and tokenize datasets
    print("Loading and tokenizing datasets...")
    train_dataset, val_dataset, _ = get_tokenized_datasets(tokenizer)

    # 3. Data collator — handles padding within each batch
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8
    )

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",        # Evaluate after every epoch
        save_strategy="epoch",        # Save checkpoint after every epoch
        load_best_model_at_end=True,  # Load best checkpoint at the end
        metric_for_best_model="eval_loss",
        greater_is_better=False,      # Lower loss is better
        logging_steps=50,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=False,                   # No fp16 on CPU
        report_to="none",             # Don't report to wandb etc.
        save_total_limit=2,           # Only keep 2 checkpoints to save disk space
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # 6. Train
    print("Starting training...")
    trainer.train()

    # 7. Save final model
    print(f"Saving final model to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="checkpoints", help="Where to save checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    train(
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )