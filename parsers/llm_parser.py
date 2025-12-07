"""
LLM-based parser using Qwen 2.5-14B-Instruct.

This parser uses a Large Language Model for semantic understanding,
combined with lightweight tokenization for SDXL alignment.
"""

import json
import re
import torch
from typing import List, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import spacy

from .base_parser import BaseParser

# Import SDXL tokenizer utilities from original code
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import get_indices


class LLMParser(BaseParser):
    """Parser using Qwen 2.5-14B for semantic understanding."""

    def __init__(self, model_path: str = None, llm_model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
        """
        Initialize LLM parser.

        Args:
            model_path: Path to SDXL model (for tokenizer alignment)
            llm_model_name: HuggingFace model name for LLM
        """
        super().__init__(model_path)
        self.llm_model_name = llm_model_name
        self.llm_model = None
        self.llm_tokenizer = None

        # Initialize SDXL tokenizer for alignment
        from transformers import AutoTokenizer as SDXLTokenizer
        self.sdxl_tokenizer = SDXLTokenizer.from_pretrained(
            model_path if model_path else "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder='tokenizer',
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # Lightweight SpaCy for tokenization only (not dependency parsing)
        self.spacy_nlp = spacy.load("en_core_web_sm")

        print(f"LLM Parser initialized with {llm_model_name}")

    def _load_llm(self):
        """Load LLM model (sequential approach - load before parsing)."""
        if self.llm_model is None:
            print(f"Loading {self.llm_model_name}...")
            print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_4bit=True,  # 4-bit quantization (~5-6GB for 14B)
            )
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)

            print(f"LLM loaded! GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    def _unload_llm(self):
        """Unload LLM to free GPU memory for SDXL."""
        if self.llm_model is not None:
            print("Unloading LLM to free memory for SDXL...")
            import gc

            del self.llm_model
            del self.llm_tokenizer
            self.llm_model = None
            self.llm_tokenizer = None

            gc.collect()
            torch.cuda.empty_cache()

            print(f"LLM unloaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    def _llm_semantic_parsing(self, prompt: str) -> Dict[str, Any]:
        """
        Use LLM to extract semantic groups from prompt.

        Args:
            prompt: Input text prompt

        Returns:
            Dictionary with semantic groups
        """
        system_prompt = """You are a semantic parser for image generation prompts.
Identify distinct semantic groups where objects bind with their attributes.

Output ONLY valid JSON in this exact format:
{
  "groups": [
    {
      "object": "main noun",
      "attributes": ["describing words"],
      "phrase": "complete phrase including determiners",
      "object_word": "exact word from prompt",
      "attribute_words": ["exact words from prompt"]
    }
  ]
}

Rules:
1. Each group has ONE main object (noun)
2. Attributes are words that describe ONLY that object
3. Use exact words from the input prompt (preserve case)
4. Include determiners (a, the) in phrase
5. For compound attributes, list each word separately in attribute_words

Examples:

Input: "a red ball and a blue cube"
Output:
{
  "groups": [
    {"object": "ball", "attributes": ["red"], "phrase": "a red ball", "object_word": "ball", "attribute_words": ["red"]},
    {"object": "cube", "attributes": ["blue"], "phrase": "a blue cube", "object_word": "cube", "attribute_words": ["blue"]}
  ]
}

Input: "a fluffy white cat wearing sunglasses"
Output:
{
  "groups": [
    {"object": "cat", "attributes": ["fluffy", "white"], "phrase": "a fluffy white cat", "object_word": "cat", "attribute_words": ["fluffy", "white"]},
    {"object": "sunglasses", "attributes": [], "phrase": "sunglasses", "object_word": "sunglasses", "attribute_words": []}
  ]
}

Input: "Van Gogh style sunset painting"
Output:
{
  "groups": [
    {"object": "painting", "attributes": ["Van Gogh style", "sunset"], "phrase": "Van Gogh style sunset painting", "object_word": "painting", "attribute_words": ["Van", "Gogh", "style", "sunset"]}
  ]
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Input: "{prompt}"'}
        ]

        text = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.llm_tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,  # Deterministic
                pad_token_id=self.llm_tokenizer.eos_token_id
            )

        response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response (after "assistant" marker)
        if "assistant" in response:
            # Split by "assistant" and take the last part
            response = response.split("assistant")[-1].strip()

        # Extract JSON from response (first valid JSON object only)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                parsed = json.loads(json_str)
                return parsed
            except json.JSONDecodeError as e:
                # Try to find the JSON more carefully
                # Look for the complete groups array
                groups_match = re.search(r'\{\s*"groups"\s*:\s*\[.*?\]\s*\}', response, re.DOTALL)
                if groups_match:
                    try:
                        parsed = json.loads(groups_match.group())
                        return parsed
                    except json.JSONDecodeError:
                        pass

                print(f"JSON parsing error: {e}")
                print(f"Extracted JSON string: {json_str[:200]}...")
                print(f"Full LLM response: {response[:500]}...")
                raise ValueError(f"LLM returned invalid JSON: {e}")
        else:
            print(f"Full LLM response: {response}")
            raise ValueError(f"No JSON found in LLM response")

    def _find_word_in_sdxl_tokens(self, word: str, sdxl_token_strings: List[str]) -> List[int]:
        """
        Find where a word appears in SDXL tokenization.
        Handles multi-wordpiece tokens (e.g., "sunglasses" -> ["sun", "##glasses"]).

        Args:
            word: Word to find
            sdxl_token_strings: SDXL tokenized strings

        Returns:
            List of token positions
        """
        word_lower = word.lower()

        # Simple case: exact match (ignoring case and </w> markers)
        for idx, token in enumerate(sdxl_token_strings):
            token_clean = token.lower().replace('</w>', '').replace('##', '')
            if token_clean == word_lower:
                return [idx]

        # Complex case: word split across multiple tokens
        for start_idx in range(len(sdxl_token_strings)):
            reconstructed = ""
            positions = []

            for offset in range(min(5, len(sdxl_token_strings) - start_idx)):
                token = sdxl_token_strings[start_idx + offset]
                token_clean = token.lower().replace('</w>', '').replace('##', '')
                reconstructed += token_clean
                positions.append(start_idx + offset)

                if reconstructed == word_lower:
                    return positions

        # Word not found - print warning
        print(f"Warning: Could not find '{word}' in SDXL tokens: {sdxl_token_strings}")
        return []

    def _align_semantic_groups(self, semantic_groups: Dict[str, Any], prompt: str) -> List[List[List[int]]]:
        """
        Align LLM's semantic groups with SDXL token positions.

        Args:
            semantic_groups: Output from LLM
            prompt: Original prompt

        Returns:
            token_indices in ToMe format: [[[noun_pos], [attr_pos]], ...]
        """
        # Get SDXL tokenization
        sdxl_tokens = self.sdxl_tokenizer(prompt)
        sdxl_token_strings = self.sdxl_tokenizer.convert_ids_to_tokens(sdxl_tokens["input_ids"])

        print(f"SDXL tokens: {sdxl_token_strings}")

        all_indices = []

        for group in semantic_groups["groups"]:
            object_word = group["object_word"]
            attribute_words = group.get("attribute_words", [])

            # Find object position in SDXL tokens
            object_positions = self._find_word_in_sdxl_tokens(object_word, sdxl_token_strings)

            # Find attribute positions
            attribute_positions = []
            for attr_word in attribute_words:
                positions = self._find_word_in_sdxl_tokens(attr_word, sdxl_token_strings)
                attribute_positions.extend(positions)

            # Only add if we found both object and attributes
            if object_positions and attribute_positions:
                all_indices.append([object_positions, attribute_positions])
                print(f"Aligned: {object_word}@{object_positions} + {attribute_words}@{attribute_positions}")
            elif object_positions and not attribute_words:
                # Object with no attributes - skip or include based on ToMe requirements
                print(f"Skipping {object_word} (no attributes)")
            else:
                print(f"Warning: Could not align group: {group}")

        return all_indices

    def parse(self, prompt: str) -> Tuple[List[List[List[int]]], List[str]]:
        """
        Parse prompt using LLM semantic understanding.

        Sequential approach:
        1. Load LLM
        2. Parse prompt
        3. Unload LLM (free memory for SDXL)
        4. Return results

        Args:
            prompt: Input text prompt

        Returns:
            Tuple of (token_indices, prompt_anchor)
        """
        print(f"\n{'='*60}")
        print(f"LLM Parsing: '{prompt}'")
        print(f"{'='*60}")

        # Load LLM
        self._load_llm()

        try:
            # Get semantic groups from LLM
            semantic_groups = self._llm_semantic_parsing(prompt)
            print(f"\nLLM identified {len(semantic_groups['groups'])} semantic groups:")
            for i, group in enumerate(semantic_groups['groups']):
                print(f"  {i+1}. {group['phrase']}")

            # Align with SDXL tokenization
            token_indices = self._align_semantic_groups(semantic_groups, prompt)

            # Extract anchor prompts
            prompt_anchor = [group["phrase"] for group in semantic_groups["groups"]]

            print(f"\nFinal output:")
            print(f"  token_indices: {token_indices}")
            print(f"  prompt_anchor: {prompt_anchor}")
            print(f"{'='*60}\n")

            return token_indices, prompt_anchor

        finally:
            # Always unload LLM (even if there's an error)
            self._unload_llm()

    def cleanup(self):
        """Clean up LLM resources."""
        self._unload_llm()
