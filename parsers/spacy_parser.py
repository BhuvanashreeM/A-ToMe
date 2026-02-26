"""
SpaCy-based parser (wraps original ToMe implementation).

This wrapper maintains backward compatibility with the original code.
"""

from typing import List, Tuple
from .base_parser import BaseParser

# Import original ToMe functions from prompt_utils
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import PromptParser


class SpaCyParser(BaseParser):
    """Parser using SpaCy dependency parsing (original ToMe approach)."""

    def __init__(self, model_path: str = None):
        """
        Initialize SpaCy parser.

        Args:
            model_path: Path to SDXL model for tokenizer
        """
        super().__init__(model_path)
        self.prompt_parser = PromptParser(model_path)
        self.nlp = None  # Lazy loaded

    def _load_spacy(self):
        """Lazy load SpaCy model."""
        if self.nlp is None:
            import en_core_web_trf
            self.nlp = en_core_web_trf.load()
            print("SpaCy model loaded")

    def parse(self, prompt: str) -> Tuple[List[List[List[int]]], List[str]]:
        """
        Parse prompt using SpaCy dependency parsing.

        Args:
            prompt: Input text prompt

        Returns:
            Tuple of (token_indices, prompt_anchor)
        """
        self._load_spacy()

        # Use original implementation
        doc = self.nlp(prompt)
        self.prompt_parser.set_doc(doc)

        token_indices = self.prompt_parser._get_indices(prompt)
        prompt_anchor = self.prompt_parser._split_prompt(doc)

        # Filter empty results
        final_idx = []
        final_prompt = []
        for i, idx in enumerate(token_indices):
            if len(idx[1]) == 0:  # Skip if no attributes
                continue
            final_idx.append(idx)
            if i < len(prompt_anchor):
                final_prompt.append(prompt_anchor[i])

        return final_idx, final_prompt

    def cleanup(self):
        """Clean up SpaCy resources."""
        # SpaCy doesn't need explicit cleanup
        pass
