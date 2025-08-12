#!/usr/bin/env python3
"""
Main training script for fine-tuning LLM with Obsidian files.
This script orchestrates the entire pipeline from parsing to training.
"""

import argparse
import logging
from pathlib import Path
import sys
import json

from obsidian_parser import ObsidianParser
from data_preprocessor import DataPreprocessor
from fine_tuner import ObsidianFineTuner, FineTuningConfig
from config import ConfigPresets, DEFAULT_OBSIDIAN_CONFIG, DEFAULT_DATASET_CONFIG, DEFAULT_MODEL_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune LLM with Obsidian files")
    
    # Data arguments
    parser.add_argument("--vault-path", type=str, default=r"c:\Users\vikto\source\MeBy2\km",
                       help="Path to Obsidian vault")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                       help="Output directory for processed datasets")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="microsoft/DialoGPT-small",
                       help="Hugging Face model name")
    parser.add_argument("--model-output-dir", type=str, default="models/obsidian-finetuned",
                       help="Directory to save fine-tuned model")
    
    # Training arguments
    parser.add_argument("--dataset-type", type=str, choices=["instruction", "conversational", "completion"],
                       default="instruction", help="Type of dataset to create")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    
    # Configuration presets
    parser.add_argument("--preset", type=str, choices=["quick_test", "production", "resource_constrained"],
                       help="Use predefined configuration preset")
    
    # Workflow control
    parser.add_argument("--skip-parsing", action="store_true", help="Skip parsing step if data already exists")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--only-parse", action="store_true", help="Only parse files, don't train")
    parser.add_argument("--only-preprocess", action="store_true", help="Only preprocess data, don't train")
    
    # Experiment tracking
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for experiment tracking")
    parser.add_argument("--wandb-project", type=str, default="obsidian-llm-finetuning",
                       help="Weights & Biases project name")
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_arguments()
    
    logger.info("Starting Obsidian LLM Fine-tuning Pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    # Apply configuration preset if specified
    if args.preset:
        logger.info(f"Applying configuration preset: {args.preset}")
        if args.preset == "quick_test":
            obsidian_config, dataset_config, model_config = ConfigPresets.quick_test()
        elif args.preset == "production":
            obsidian_config, dataset_config, model_config = ConfigPresets.production()
        elif args.preset == "resource_constrained":
            obsidian_config, dataset_config, model_config = ConfigPresets.resource_constrained()
    else:
        # Use default configurations
        obsidian_config = DEFAULT_OBSIDIAN_CONFIG
        dataset_config = DEFAULT_DATASET_CONFIG
        model_config = DEFAULT_MODEL_CONFIG
    
    # Override with command line arguments
    obsidian_config.vault_path = args.vault_path
    dataset_config.output_dir = args.output_dir
    model_config.model_name = args.model_name
    model_config.output_dir = args.model_output_dir
    model_config.batch_size = args.batch_size
    model_config.num_epochs = args.epochs
    model_config.learning_rate = args.learning_rate
    model_config.max_length = args.max_length
    model_config.use_wandb = args.use_wandb
    model_config.wandb_project = args.wandb_project
    
    # Create output directories
    Path(dataset_config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(model_config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Parse Obsidian files
    parsed_files = None
    parsed_files_path = Path(dataset_config.output_dir) / "parsed_files.json"
    
    if not args.skip_parsing or not parsed_files_path.exists():
        logger.info("Step 1: Parsing Obsidian files...")
        parser = ObsidianParser(obsidian_config.vault_path)
        parsed_files = parser.parse_vault(obsidian_config.file_extensions)
        
        # Save parsed files for reuse
        with open(parsed_files_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_files, f, ensure_ascii=False, indent=2)
        
        # Print statistics
        stats = parser.get_statistics(parsed_files)
        logger.info(f"Parsed {stats['total_files']} files with {stats['total_words']:,} total words")
        for category, data in stats['categories'].items():
            logger.info(f"  {category}: {data['count']} files, {data['words']:,} words")
    else:
        logger.info("Step 1: Loading previously parsed files...")
        with open(parsed_files_path, 'r', encoding='utf-8') as f:
            parsed_files = json.load(f)
        logger.info(f"Loaded {len(parsed_files)} parsed files")
    
    if args.only_parse:
        logger.info("Only parsing requested. Exiting.")
        return
    
    # Step 2: Preprocess data for training
    dataset_path = Path(dataset_config.output_dir) / f"{args.dataset_type}_dataset.jsonl"
    
    if not args.skip_preprocessing or not dataset_path.exists():
        logger.info(f"Step 2: Creating {args.dataset_type} dataset...")
        preprocessor = DataPreprocessor(
            min_length=dataset_config.min_example_length,
            max_length=dataset_config.max_example_length
        )
        
        if args.dataset_type == "instruction":
            dataset = preprocessor.create_instruction_dataset(parsed_files)
        elif args.dataset_type == "conversational":
            dataset = preprocessor.create_conversational_dataset(parsed_files)
        elif args.dataset_type == "completion":
            dataset = preprocessor.create_completion_dataset(parsed_files)
        
        # Split dataset
        train_data, val_data, test_data = preprocessor.split_dataset(
            dataset, 
            train_ratio=dataset_config.train_ratio,
            val_ratio=dataset_config.val_ratio
        )
        
        # Save datasets
        preprocessor.save_dataset(train_data, dataset_path.parent / f"train_{args.dataset_type}_dataset.jsonl")
        preprocessor.save_dataset(val_data, dataset_path.parent / f"val_{args.dataset_type}_dataset.jsonl")
        preprocessor.save_dataset(test_data, dataset_path.parent / f"test_{args.dataset_type}_dataset.jsonl")
        preprocessor.save_dataset(dataset, dataset_path)  # Full dataset
        
        logger.info(f"Created datasets: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    else:
        logger.info("Step 2: Using existing preprocessed dataset...")
    
    if args.only_preprocess:
        logger.info("Only preprocessing requested. Exiting.")
        return
    
    # Step 3: Fine-tune the model
    logger.info("Step 3: Fine-tuning the model...")
    
    # Create fine-tuning configuration
    fine_tuning_config = FineTuningConfig(
        model_name=model_config.model_name,
        output_dir=model_config.output_dir,
        max_length=model_config.max_length,
        batch_size=model_config.batch_size,
        gradient_accumulation_steps=model_config.gradient_accumulation_steps,
        learning_rate=model_config.learning_rate,
        num_epochs=model_config.num_epochs,
        warmup_steps=model_config.warmup_steps,
        logging_steps=model_config.logging_steps,
        save_steps=model_config.save_steps,
        eval_steps=model_config.eval_steps,
        save_total_limit=model_config.save_total_limit,
        load_best_model_at_end=model_config.load_best_model_at_end,
        metric_for_best_model=model_config.metric_for_best_model,
        greater_is_better=model_config.greater_is_better,
        use_wandb=model_config.use_wandb,
        wandb_project=model_config.wandb_project
    )
    
    # Create and run fine-tuner
    fine_tuner = ObsidianFineTuner(fine_tuning_config)
    trainer = fine_tuner.fine_tune(
        str(dataset_path.parent / f"train_{args.dataset_type}_dataset.jsonl"),
        dataset_type=args.dataset_type,
        validation_split=0.0  # We already have separate validation data
    )
    
    # Step 4: Test the model
    logger.info("Step 4: Testing the fine-tuned model...")
    
    # Test prompts based on dataset type
    if args.dataset_type == "instruction":
        test_prompts = [
            "### Instruction:\nWhat is AI?\n\n### Response:\n",
            "### Instruction:\nExplain the concept of machine learning.\n\n### Response:\n",
            "### Instruction:\nDescribe the PARA method for organizing notes.\n\n### Response:\n"
        ]
    elif args.dataset_type == "conversational":
        test_prompts = [
            "Human: What is artificial intelligence?\nAssistant:",
            "Human: How does machine learning work?\nAssistant:",
            "Human: Tell me about project management.\nAssistant:"
        ]
    else:  # completion
        test_prompts = [
            "Artificial intelligence is",
            "The main benefit of using",
            "In project management, it's important to"
        ]
    
    logger.info("Sample generations:")
    for i, prompt in enumerate(test_prompts):
        try:
            generated = fine_tuner.generate_text(prompt, max_length=100)
            logger.info(f"Prompt {i+1}: {prompt}")
            logger.info(f"Generated: {generated}")
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"Error generating text for prompt {i+1}: {e}")
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Model saved to: {model_config.output_dir}")
    logger.info(f"Datasets saved to: {dataset_config.output_dir}")


if __name__ == "__main__":
    main()
