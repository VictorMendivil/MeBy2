#!/usr/bin/env python3
"""
Inference script for interacting with the fine-tuned Obsidian LLM.
"""

import argparse
import torch
from pathlib import Path
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import DEFAULT_GENERATION_CONFIG

logger = logging.getLogger(__name__)


class ObsidianLLMInference:
    """Inference class for the fine-tuned Obsidian LLM."""
    
    def __init__(self, model_path: str):
        """Initialize the inference engine."""
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def generate_response(self, prompt: str, max_length: int = None, 
                         temperature: float = None, top_p: float = None,
                         top_k: int = None, repetition_penalty: float = None) -> str:
        """Generate a response to the given prompt."""
        # Use default config values if not provided
        config = DEFAULT_GENERATION_CONFIG
        max_length = max_length or config.max_length
        temperature = temperature or config.temperature
        top_p = top_p or config.top_p
        top_k = top_k or config.top_k
        repetition_penalty = repetition_penalty or config.repetition_penalty
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=config.do_sample,
                num_return_sequences=config.num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # Decode and clean up
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def ask_question(self, question: str) -> str:
        """Ask a question using the instruction format."""
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        return self.generate_response(prompt)
    
    def continue_text(self, text: str) -> str:
        """Continue the given text."""
        return self.generate_response(text)
    
    def chat(self, message: str, conversation_history: list = None) -> str:
        """Have a conversation using the conversational format."""
        if conversation_history is None:
            conversation_history = []
        
        # Build conversation context
        context = ""
        for turn in conversation_history:
            context += f"Human: {turn['human']}\nAssistant: {turn['assistant']}\n"
        
        # Add current message
        context += f"Human: {message}\nAssistant:"
        
        response = self.generate_response(context)
        return response


def interactive_mode(inference_engine: ObsidianLLMInference, mode: str = "instruction"):
    """Run interactive mode for chatting with the model."""
    print(f"\nObsidian LLM Interactive Mode ({mode})")
    print("Type 'quit' to exit, 'clear' to clear conversation history")
    print("-" * 50)
    
    conversation_history = []
    
    while True:
        try:
            if mode == "instruction":
                user_input = input("\nAsk a question: ")
            elif mode == "chat":
                user_input = input("\nYou: ")
            else:  # completion
                user_input = input("\nStart text: ")
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                conversation_history = []
                print("Conversation history cleared!")
                continue
            
            if not user_input.strip():
                continue
            
            print("Thinking...")
            
            if mode == "instruction":
                response = inference_engine.ask_question(user_input)
            elif mode == "chat":
                response = inference_engine.chat(user_input, conversation_history)
                conversation_history.append({"human": user_input, "assistant": response})
            else:  # completion
                response = inference_engine.continue_text(user_input)
            
            print(f"Response: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(description="Interact with fine-tuned Obsidian LLM")
    
    parser.add_argument("--model-path", type=str, default="models/obsidian-finetuned",
                       help="Path to fine-tuned model")
    parser.add_argument("--mode", type=str, choices=["instruction", "chat", "completion"],
                       default="instruction", help="Interaction mode")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate response for")
    parser.add_argument("--max-length", type=int, default=200, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    # Load model
    try:
        inference_engine = ObsidianLLMInference(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have trained a model first using train_obsidian_llm.py")
        return
    
    if args.prompt:
        # Single prompt mode
        print(f"Prompt: {args.prompt}")
        print("Generating...")
        
        if args.mode == "instruction":
            response = inference_engine.ask_question(args.prompt)
        elif args.mode == "chat":
            response = inference_engine.chat(args.prompt)
        else:  # completion
            response = inference_engine.continue_text(args.prompt)
        
        print(f"Response: {response}")
    else:
        # Interactive mode
        interactive_mode(inference_engine, args.mode)


if __name__ == "__main__":
    main()
