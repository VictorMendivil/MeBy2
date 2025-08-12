#!/usr/bin/env python3
"""
Obsidian file parser for extracting and preprocessing content for LLM fine-tuning.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObsidianParser:
    """Parser for Obsidian markdown files."""
    
    def __init__(self, vault_path: str):
        """Initialize the parser with the vault path."""
        self.vault_path = Path(vault_path)
        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {vault_path}")
    
    def extract_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata from markdown content."""
        metadata = {}
        
        # Extract tags
        tags = re.findall(r'#([a-zA-Z0-9_/\-]+)', content)
        if tags:
            metadata['tags'] = ', '.join(tags)
        
        # Extract links
        links = re.findall(r'\[\[([^\]]+)\]\]', content)
        if links:
            metadata['links'] = ', '.join(links)
        
        # Extract headings
        headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        if headings:
            metadata['headings'] = ', '.join(headings)
        
        return metadata
    
    def clean_content(self, content: str) -> str:
        """Clean markdown content for training."""
        # Remove YAML frontmatter
        content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
        
        # Convert Obsidian links to plain text
        content = re.sub(r'\[\[([^\]]+)\]\]', r'\1', content)
        
        # Convert markdown links to plain text
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        
        # Remove task syntax
        content = re.sub(r'```tasks.*?```', '', content, flags=re.DOTALL)
        
        # Clean up excessive whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = content.strip()
        
        return content
    
    def parse_file(self, file_path: Path) -> Optional[Dict[str, str]]:
        """Parse a single Obsidian file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            if not raw_content.strip():
                return None
            
            # Extract metadata
            metadata = self.extract_metadata(raw_content)
            
            # Clean content
            clean_content = self.clean_content(raw_content)
            
            if len(clean_content) < 10:  # Skip very short files
                return None
            
            # Determine category from path
            relative_path = file_path.relative_to(self.vault_path)
            category = str(relative_path.parts[0]) if relative_path.parts else "unknown"
            
            return {
                'file_path': str(file_path),
                'relative_path': str(relative_path),
                'category': category,
                'title': file_path.stem,
                'content': clean_content,
                'raw_content': raw_content,
                'word_count': len(clean_content.split()),
                **metadata
            }
            
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {e}")
            return None
    
    def parse_vault(self, file_extensions: List[str] = ['.md']) -> List[Dict[str, str]]:
        """Parse all files in the vault."""
        parsed_files = []
        
        for ext in file_extensions:
            for file_path in self.vault_path.rglob(f'*{ext}'):
                parsed_file = self.parse_file(file_path)
                if parsed_file:
                    parsed_files.append(parsed_file)
        
        logger.info(f"Parsed {len(parsed_files)} files from vault")
        return parsed_files
    
    def get_statistics(self, parsed_files: List[Dict[str, str]]) -> Dict[str, int]:
        """Get statistics about the parsed files."""
        stats = {
            'total_files': len(parsed_files),
            'total_words': sum(f['word_count'] for f in parsed_files),
            'categories': {}
        }
        
        for file_data in parsed_files:
            category = file_data['category']
            if category not in stats['categories']:
                stats['categories'][category] = {'count': 0, 'words': 0}
            stats['categories'][category]['count'] += 1
            stats['categories'][category]['words'] += file_data['word_count']
        
        return stats


def main():
    """Example usage of the ObsidianParser."""
    vault_path = r"c:\Users\vikto\source\MeBy2\km"
    
    parser = ObsidianParser(vault_path)
    parsed_files = parser.parse_vault()
    
    # Print statistics
    stats = parser.get_statistics(parsed_files)
    print(f"Vault Statistics:")
    print(f"Total files: {stats['total_files']}")
    print(f"Total words: {stats['total_words']:,}")
    print(f"\nBy category:")
    for category, data in stats['categories'].items():
        print(f"  {category}: {data['count']} files, {data['words']:,} words")
    
    # Show sample content
    if parsed_files:
        print(f"\nSample file content:")
        sample = parsed_files[0]
        print(f"Title: {sample['title']}")
        print(f"Category: {sample['category']}")
        print(f"Content preview: {sample['content'][:200]}...")


if __name__ == "__main__":
    main()
