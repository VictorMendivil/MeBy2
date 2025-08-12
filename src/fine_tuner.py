#!/usr/bin/env python3
"""
Fine-tuning script for LLMs using Hugging Face Transformers with Obsidian data.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass, field

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset
import wandb

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""
    model_name: str = "microsoft/DialoGPT-medium"  # Good for conversational fine-tuning
    output_dir: str = "models/obsidian-finetuned"
    max_length: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    use_wandb: bool = False
    wandb_project: str = "obsidian-llm-finetuning"


class ObsidianFineTuner:
    """Fine-tuner for LLMs using Obsidian data."""
    
    def __init__(self, config: FineTuningConfig):
        """Initialize the fine-tuner with configuration."""
        self.config = config
        self.tokenizer = None
        self.model = None
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(project=config.wandb_project)
    
    def load_model_and_tokenizer(self):
        """Load the pre-trained model and tokenizer."""
        logger.info(f"Loading model and tokenizer: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to device
        self.model.to(self.device)
        
        logger.info(f"Model loaded with {self.model.num_parameters():,} parameters")
    
    def prepare_instruction_dataset(self, dataset_path: str) -> Dataset:
        """Prepare instruction dataset for training."""
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        # Format for instruction following
        formatted_data = []
        for item in data:
            # Create instruction-following format
            text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            formatted_data.append({'text': text})
        
        return Dataset.from_list(formatted_data)
    
    def prepare_conversational_dataset(self, dataset_path: str) -> Dataset:
        """Prepare conversational dataset for training."""
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        # Format for conversation
        formatted_data = []
        for item in data:
            messages = item['messages']
            conversation = ""
            for msg in messages:
                if msg['role'] == 'user':
                    conversation += f"Human: {msg['content']}\n"
                else:
                    conversation += f"Assistant: {msg['content']}\n"
            formatted_data.append({'text': conversation.strip()})
        
        return Dataset.from_list(formatted_data)
    
    def prepare_completion_dataset(self, dataset_path: str) -> Dataset:
        """Prepare completion dataset for training."""
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        # Format for completion
        formatted_data = []
        for item in data:
            text = f"{item['prompt']}{item['completion']}"
            formatted_data.append({'text': text})
        
        return Dataset.from_list(formatted_data)
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset."""
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].clone()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def create_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> Trainer:
        """Create the Trainer object."""
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            report_to="wandb" if self.config.use_wandb else None,
            fp16=torch.cuda.is_available(),  # Use mixed precision if CUDA available
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else None,
        )
        
        return trainer
    
    def fine_tune(self, dataset_path: str, dataset_type: str = "instruction", 
                  validation_split: float = 0.1):
        """Fine-tune the model on the dataset."""
        logger.info(f"Starting fine-tuning with {dataset_type} dataset")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Prepare dataset
        if dataset_type == "instruction":
            dataset = self.prepare_instruction_dataset(dataset_path)
        elif dataset_type == "conversational":
            dataset = self.prepare_conversational_dataset(dataset_path)
        elif dataset_type == "completion":
            dataset = self.prepare_completion_dataset(dataset_path)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        logger.info(f"Loaded dataset with {len(dataset)} examples")
        
        # Split dataset
        if validation_split > 0:
            split_dataset = dataset.train_test_split(test_size=validation_split)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
        else:
            train_dataset = dataset
            eval_dataset = None
        
        # Tokenize datasets
        train_dataset = self.tokenize_dataset(train_dataset)
        if eval_dataset:
            eval_dataset = self.tokenize_dataset(eval_dataset)
        
        # Create trainer
        trainer = self.create_trainer(train_dataset, eval_dataset)
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info(f"Training completed. Model saved to {self.config.output_dir}")
        
        return trainer
    
    def generate_text(self, prompt: str, max_length: int = 200, 
                     temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate text using the fine-tuned model."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output
        generated_text = generated_text[len(prompt):].strip()
        
        return generated_text


def main():
    """Example usage of the ObsidianFineTuner."""
    # Configuration
    config = FineTuningConfig(
        model_name="microsoft/DialoGPT-small",  # Smaller model for testing
        output_dir="models/obsidian-finetuned",
        batch_size=2,  # Smaller batch size for limited resources
        num_epochs=1,  # Quick test
        max_length=256,
        use_wandb=False
    )
    
    # Create fine-tuner
    fine_tuner = ObsidianFineTuner(config)
    
    # Fine-tune on instruction dataset
    dataset_path = "data/processed/instruction_dataset.jsonl"
    if Path(dataset_path).exists():
        trainer = fine_tuner.fine_tune(dataset_path, dataset_type="instruction")
        
        # Test generation
        prompt = "### Instruction:\nWhat is AI?\n\n### Response:\n"
        generated = fine_tuner.generate_text(prompt)
        print(f"Generated text: {generated}")
    else:
        print(f"Dataset not found: {dataset_path}")
        print("Please run the data preprocessor first to create the dataset.")


if __name__ == "__main__":
    main()
