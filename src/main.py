#!/usr/bin/env python3
"""
Main module for MeBy2 project - Obsidian LLM Fine-tuning.

This is the main entry point for the Obsidian LLM fine-tuning project.
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from train_obsidian_llm import main as train_main
from inference import main as inference_main
from obsidian_parser import main as parser_main
from data_preprocessor import main as preprocessor_main


def show_help():
    """Show help information about available commands."""
    help_text = """
🧠 Obsidian LLM Fine-tuning Project
==================================

Available commands:

1. parse           - Parse Obsidian vault and extract content
2. preprocess      - Create training datasets from parsed content
3. train           - Train/fine-tune an LLM on your Obsidian data
4. inference       - Interact with the trained model
5. pipeline        - Run the complete training pipeline
6. help            - Show this help message

Examples:
---------
python main.py parse                    # Parse Obsidian files
python main.py preprocess              # Create training datasets
python main.py train --preset quick_test  # Quick training test
python main.py inference --mode chat   # Chat with trained model
python main.py pipeline --preset production  # Full pipeline

For detailed options for each command, use:
python main.py <command> --help

Quick Start:
-----------
1. First time setup (quick test):
   python main.py pipeline --preset quick_test

2. Production training:
   python main.py pipeline --preset production

3. Chat with your trained model:
   python main.py inference --mode chat
"""
    print(help_text)


def main():
    """Main function - entry point of the application."""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    # Remove the command from sys.argv so subcommands can parse their own args
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    try:
        if command == "parse":
            print("Running Obsidian parser...")
            parser_main()
        
        elif command == "preprocess":
            print("Running data preprocessor...")
            preprocessor_main()
        
        elif command in ["train", "pipeline"]:
            print("Running training pipeline...")
            train_main()
        
        elif command == "inference":
            print("Starting inference mode...")
            inference_main()
        
        elif command == "help":
            show_help()
        
        else:
            print(f"Unknown command: {command}")
            print("Use 'python main.py help' to see available commands")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
