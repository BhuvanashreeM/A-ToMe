"""
Base parser interface for ToMe prompt parsing.

"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseParser(ABC):
    """Abstract base class for prompt parsers."""

    def __init__(self, model_path: str = None):
        """
        Initialize parser.

        Args:
            model_path: Path to SDXL model (for tokenizer alignment)
        """
        self.model_path = model_path

    @abstractmethod
    def parse(self, prompt: str) -> Tuple[List[List[List[int]]], List[str]]:
        """
        Parse a prompt

        Args:
            prompt: Input text prompt (example "a white cat and a black dog")

        Returns:
            Tuple of:
                - token_indices: [[[noun_positions], [attribute_positions]], ...]
                  Example: [[[3], [2]], [[7], [6]]]
                - prompt_anchor: List of complete phrases
                  Example: ["a white cat", "a black dog"]
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources (e.g., unload models from GPU)."""
        pass

