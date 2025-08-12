#!/usr/bin/env python3
"""
Configuration file for Obsidian LLM fine-tuning project.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ObsidianConfig:
    """Configuration for Obsidian parsing."""
    vault_path: str = r"c:\Users\vikto\source\MeBy2\km"
    file_extensions: List[str] = None
    min_content_length: int = 50
    max_content_length: int = 2048
    exclude_patterns: List[str] = None
    
    def __post_init__(self):
        if self.file_extensions is None:
            self.file_extensions = ['.md']
        if self.exclude_patterns is None:
            self.exclude_patterns = ['*.tmp', '*.bak', '.obsidian/*']


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""
    output_dir: str = "data/processed"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    min_example_length: int = 20
    max_example_length: int = 1024
    include_categories: Optional[List[str]] = None
    exclude_categories: Optional[List[str]] = None
    
    def __post_init__(self):
        # Ensure ratios sum to 1.0
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")


@dataclass
class ModelConfig:
    """Configuration for model selection and training."""
    # Model options for different use cases
    SMALL_MODELS = [
        "microsoft/DialoGPT-small",
        "distilgpt2",
        "gpt2"
    ]
    
    MEDIUM_MODELS = [
        "microsoft/DialoGPT-medium",
        "gpt2-medium"
    ]
    
    LARGE_MODELS = [
        "microsoft/DialoGPT-large",
        "gpt2-large"
    ]
    
    # Default model selection
    model_name: str = "microsoft/DialoGPT-small"
    output_dir: str = "models/obsidian-finetuned"
    
    # Training hyperparameters
    max_length: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    
    # Early stopping
    early_stopping_patience: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Hardware optimization
    use_fp16: bool = True
    use_gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "obsidian-llm-finetuning"
    experiment_name: Optional[str] = None


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 1


# Predefined configurations for different scenarios
class ConfigPresets:
    """Predefined configuration presets for common scenarios."""
    
    @staticmethod
    def quick_test() -> tuple:
        """Configuration for quick testing with minimal resources."""
        obsidian_config = ObsidianConfig(
            min_content_length=20,
            max_content_length=512
        )
        
        dataset_config = DatasetConfig(
            min_example_length=10,
            max_example_length=256
        )
        
        model_config = ModelConfig(
            model_name="distilgpt2",
            max_length=256,
            batch_size=2,
            num_epochs=1,
            save_steps=100,
            eval_steps=100
        )
        
        return obsidian_config, dataset_config, model_config
    
    @staticmethod
    def production() -> tuple:
        """Configuration for production-quality fine-tuning."""
        obsidian_config = ObsidianConfig()
        
        dataset_config = DatasetConfig()
        
        model_config = ModelConfig(
            model_name="microsoft/DialoGPT-medium",
            max_length=1024,
            batch_size=8,
            gradient_accumulation_steps=2,
            num_epochs=5,
            use_wandb=True
        )
        
        return obsidian_config, dataset_config, model_config
    
    @staticmethod
    def resource_constrained() -> tuple:
        """Configuration for systems with limited resources."""
        obsidian_config = ObsidianConfig(
            max_content_length=512
        )
        
        dataset_config = DatasetConfig(
            max_example_length=512
        )
        
        model_config = ModelConfig(
            model_name="distilgpt2",
            max_length=512,
            batch_size=1,
            gradient_accumulation_steps=8,
            use_gradient_checkpointing=True,
            dataloader_num_workers=0
        )
        
        return obsidian_config, dataset_config, model_config


# Default configuration
DEFAULT_OBSIDIAN_CONFIG = ObsidianConfig()
DEFAULT_DATASET_CONFIG = DatasetConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_GENERATION_CONFIG = GenerationConfig()
