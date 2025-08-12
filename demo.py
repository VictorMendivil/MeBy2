#!/usr/bin/env python3
"""
Demo script for the Obsidian LLM Fine-tuning Project.
This script demonstrates the complete workflow from parsing to training.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from obsidian_parser import ObsidianParser
from data_preprocessor import DataPreprocessor
from fine_tuner import ObsidianFineTuner, FineTuningConfig
from config import ConfigPresets


def demo_quick_test():
    """Run a quick demonstration of the complete pipeline."""
    print("=" * 60)
    print("OBSIDIAN LLM FINE-TUNING DEMO - QUICK TEST")
    print("=" * 60)
    
    # Get configuration preset for quick testing
    obsidian_config, dataset_config, model_config = ConfigPresets.quick_test()
    
    print(f"\n1. PARSING OBSIDIAN VAULT")
    print(f"   Vault path: {obsidian_config.vault_path}")
    
    # Parse Obsidian files
    parser = ObsidianParser(obsidian_config.vault_path)
    parsed_files = parser.parse_vault()
    
    # Show statistics
    stats = parser.get_statistics(parsed_files)
    print(f"   Parsed {stats['total_files']} files with {stats['total_words']:,} total words")
    
    print(f"\n   Categories breakdown:")
    for category, data in stats['categories'].items():
        print(f"   - {category}: {data['count']} files, {data['words']:,} words")
    
    print(f"\n2. CREATING TRAINING DATASETS")
    
    # Create preprocessor
    preprocessor = DataPreprocessor(
        min_length=dataset_config.min_example_length,
        max_length=dataset_config.max_example_length
    )
    
    # Create instruction dataset
    instruction_dataset = preprocessor.create_instruction_dataset(parsed_files)
    print(f"   Created {len(instruction_dataset)} instruction examples")
    
    # Show sample examples
    print(f"\n   Sample instruction examples:")
    for i, example in enumerate(instruction_dataset[:3]):
        print(f"   Example {i+1}:")
        print(f"   Q: {example['instruction']}")
        print(f"   A: {example['output'][:100]}...")
        print()
    
    print(f"\n3. MODEL TRAINING SETUP")
    print(f"   Model: {model_config.model_name}")
    print(f"   Max length: {model_config.max_length}")
    print(f"   Batch size: {model_config.batch_size}")
    print(f"   Epochs: {model_config.num_epochs}")
    
    # Save sample dataset for training
    output_dir = Path("data/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Take a small sample for quick demo
    sample_dataset = instruction_dataset[:20]  # Just 20 examples for demo
    preprocessor.save_dataset(sample_dataset, output_dir / "demo_dataset.jsonl")
    
    print(f"\n   Sample dataset saved to: {output_dir / 'demo_dataset.jsonl'}")
    print(f"   Dataset size: {len(sample_dataset)} examples")
    
    print(f"\n4. NEXT STEPS")
    print(f"   To run actual training:")
    print(f"   python src/main.py pipeline --preset quick_test")
    print(f"   ")
    print(f"   To train with your full dataset:")
    print(f"   python src/main.py pipeline --preset production")
    print(f"   ")
    print(f"   To chat with a trained model:")
    print(f"   python src/main.py inference --mode chat")
    
    print(f"\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)


def show_project_summary():
    """Show a summary of the complete project."""
    print("\n" + "=" * 60)
    print("PROJECT SUMMARY: OBSIDIAN LLM FINE-TUNING")
    print("=" * 60)
    
    print("\nWHAT THIS PROJECT DOES:")
    print("- Parses your Obsidian markdown files")
    print("- Extracts and cleans content for AI training")
    print("- Creates multiple types of training datasets")
    print("- Fine-tunes language models on your personal knowledge")
    print("- Provides interactive chat interface with your trained model")
    
    print("\nKEY COMPONENTS:")
    print("1. obsidian_parser.py - Extracts content from Obsidian vault")
    print("2. data_preprocessor.py - Creates training datasets")
    print("3. fine_tuner.py - Handles model training with Hugging Face")
    print("4. inference.py - Interactive chat with trained models")
    print("5. train_obsidian_llm.py - Complete training pipeline")
    print("6. config.py - Configuration presets and settings")
    
    print("\nDATASET TYPES:")
    print("- Instruction: Question-answer format")
    print("- Conversational: Chat-style interactions")
    print("- Completion: Text completion tasks")
    
    print("\nSUPPORTED MODELS:")
    print("- GPT-2 variants (small, medium, large)")
    print("- DialoGPT variants (small, medium, large)")
    print("- DistilGPT-2 (lightweight option)")
    
    print("\nCONFIGURATION PRESETS:")
    print("- quick_test: Fast training for testing")
    print("- production: Full training for best results")
    print("- resource_constrained: Optimized for limited hardware")


def main():
    """Main demo function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        show_project_summary()
        return
    
    try:
        demo_quick_test()
        show_project_summary()
    except Exception as e:
        print(f"Demo error: {e}")
        print("Make sure you have:")
        print("1. Installed all dependencies: pip install -r requirements.txt")
        print("2. Your Obsidian vault is accessible")
        print("3. Python environment is properly set up")


if __name__ == "__main__":
    main()
