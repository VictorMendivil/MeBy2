#!/usr/bin/env python3
"""
Data preprocessor for converting Obsidian content into training datasets for LLM fine-tuning.
"""

import json
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from obsidian_parser import ObsidianParser

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocessor for creating training datasets from Obsidian content."""
    
    def __init__(self, min_length: int = 50, max_length: int = 2048):
        """Initialize the preprocessor with length constraints."""
        self.min_length = min_length
        self.max_length = max_length
    
    def create_instruction_dataset(self, parsed_files: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Create instruction-following dataset from Obsidian content."""
        dataset = []
        
        for file_data in parsed_files:
            content = file_data['content']
            title = file_data['title']
            category = file_data['category']
            
            if len(content.split()) < self.min_length:
                continue
            
            # Create different types of instruction-response pairs
            instructions = self._generate_instructions(title, content, category)
            
            for instruction, response in instructions:
                if len(response.split()) >= self.min_length:
                    dataset.append({
                        'instruction': instruction,
                        'input': '',
                        'output': response,
                        'source_file': file_data['relative_path'],
                        'category': category
                    })
        
        return dataset
    
    def _generate_instructions(self, title: str, content: str, category: str) -> List[Tuple[str, str]]:
        """Generate various instruction-response pairs from content."""
        instructions = []
        
        # Summarization tasks
        if len(content.split()) > 100:
            instructions.append((
                f"Summarize the key points about {title}.",
                self._create_summary(content)
            ))
        
        # Question-answering tasks
        instructions.append((
            f"What is {title}?",
            content
        ))
        
        # Category-specific instructions
        if category == "3-Resources":
            instructions.append((
                f"Explain the concept of {title} in detail.",
                content
            ))
        elif category == "0-Journal":
            instructions.append((
                f"Describe the activities and thoughts from this journal entry.",
                content
            ))
        elif category == "1-Projects":
            instructions.append((
                f"Provide information about the project: {title}.",
                content
            ))
        
        # Knowledge extraction
        if "tags" in content or "#" in content:
            instructions.append((
                f"What are the main topics and concepts related to {title}?",
                content
            ))
        
        return instructions
    
    def _create_summary(self, content: str) -> str:
        """Create a summary of the content (simplified version)."""
        sentences = content.split('.')
        # Take first few sentences as summary
        summary_sentences = sentences[:3] if len(sentences) > 3 else sentences
        return '. '.join(summary_sentences).strip() + '.'
    
    def create_completion_dataset(self, parsed_files: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Create text completion dataset from Obsidian content."""
        dataset = []
        
        for file_data in parsed_files:
            content = file_data['content']
            
            if len(content.split()) < self.min_length * 2:
                continue
            
            # Split content into prompt and completion
            words = content.split()
            split_point = len(words) // 2
            
            prompt = ' '.join(words[:split_point])
            completion = ' '.join(words[split_point:])
            
            if len(completion.split()) >= self.min_length:
                dataset.append({
                    'prompt': prompt,
                    'completion': completion,
                    'source_file': file_data['relative_path'],
                    'category': file_data['category']
                })
        
        return dataset
    
    def create_conversational_dataset(self, parsed_files: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Create conversational dataset from Obsidian content."""
        dataset = []
        
        for file_data in parsed_files:
            content = file_data['content']
            title = file_data['title']
            
            if len(content.split()) < self.min_length:
                continue
            
            # Create conversational pairs
            conversations = [
                {
                    'messages': [
                        {'role': 'user', 'content': f"Tell me about {title}"},
                        {'role': 'assistant', 'content': content}
                    ],
                    'source_file': file_data['relative_path'],
                    'category': file_data['category']
                },
                {
                    'messages': [
                        {'role': 'user', 'content': f"Can you explain {title}?"},
                        {'role': 'assistant', 'content': content}
                    ],
                    'source_file': file_data['relative_path'],
                    'category': file_data['category']
                }
            ]
            
            dataset.extend(conversations)
        
        return dataset
    
    def filter_by_category(self, dataset: List[Dict], categories: List[str]) -> List[Dict]:
        """Filter dataset by specific categories."""
        return [item for item in dataset if item.get('category') in categories]
    
    def split_dataset(self, dataset: List[Dict], train_ratio: float = 0.8, 
                     val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train, validation, and test sets."""
        random.shuffle(dataset)
        
        total = len(dataset)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_data = dataset[:train_size]
        val_data = dataset[train_size:train_size + val_size]
        test_data = dataset[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def save_dataset(self, dataset: List[Dict], output_path: str, format: str = 'jsonl'):
        """Save dataset to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(dataset)} items to {output_path}")


def main():
    """Example usage of the DataPreprocessor."""
    # Parse Obsidian vault
    vault_path = r"c:\Users\vikto\source\MeBy2\km"
    parser = ObsidianParser(vault_path)
    parsed_files = parser.parse_vault()
    
    # Create preprocessor
    preprocessor = DataPreprocessor(min_length=20, max_length=1024)
    
    # Create different types of datasets
    print("Creating instruction dataset...")
    instruction_dataset = preprocessor.create_instruction_dataset(parsed_files)
    
    print("Creating completion dataset...")
    completion_dataset = preprocessor.create_completion_dataset(parsed_files)
    
    print("Creating conversational dataset...")
    conversational_dataset = preprocessor.create_conversational_dataset(parsed_files)
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Instruction dataset: {len(instruction_dataset)} examples")
    print(f"Completion dataset: {len(completion_dataset)} examples")
    print(f"Conversational dataset: {len(conversational_dataset)} examples")
    
    # Save datasets
    output_dir = Path("data/processed")
    preprocessor.save_dataset(instruction_dataset, output_dir / "instruction_dataset.jsonl")
    preprocessor.save_dataset(completion_dataset, output_dir / "completion_dataset.jsonl")
    preprocessor.save_dataset(conversational_dataset, output_dir / "conversational_dataset.jsonl")
    
    # Show sample data
    if instruction_dataset:
        print(f"\nSample instruction example:")
        sample = instruction_dataset[0]
        print(f"Instruction: {sample['instruction']}")
        print(f"Output: {sample['output'][:200]}...")


if __name__ == "__main__":
    main()
