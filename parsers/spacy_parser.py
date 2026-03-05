"""
SpaCy-based parser orig
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
        Parse prompt using SpaCy

        Args:
            prompt: Input text\prompt

        Returns:
            Tuple of (token_indices and prompt_anchor)
        """
        self._load_spacy()

        # Use original implementation
        doc = self.nlp(prompt)
        self.prompt_parser.set_doc(doc)

        tokenindices = self.prompt_parser._get_indices(prompt)
        panchor = self.prompt_parser._split_prompt(doc)

        # Filter empty results
        fidx = []
        fprompt = []
        for i, idx in enumerate(tokenindices):
            if len(idx[1]) == 0:  # Skip if no attributes
                continue
            fidx.append(idx)
            if i < len(panchor):
                fprompt.append(panchor[i])

        return fidx, fprompt

    def cleanup(self):
        """Clean up SpaCy resources."""
        # SpaCy doesn't need explicit cleanup
        pass

