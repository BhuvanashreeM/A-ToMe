"""
LLM-based parser using Qwen 2.5-14B-Instruct.

This parser uses a Large Language Model for semantic understanding,
combined with lightweight tokenization for SDXL alignment.

TODO: maybe switch to a smaller model? 14B is kinda overkill
      but the quality is nice so idk

Author: [redacted]
Last updated: whenever i last touched this lol
"""

import json
import re
import torch
from typing import List, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import spacy

from .base_parser import BaseParser

# Import SDXL tokenizer utilities from original code
# this is ugly but it works, don't @ me
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import get_indices  # not sure if we even use this anymore?


class LLMParser(BaseParser):
    """Parser using Qwen 2.5-14B for semantic understanding."""

    def __init__(self, modelpath: str = None, llmname: str = "Qwen/Qwen2.5-14B-Instruct"):
        """
        Initialize LLM parser.

        Args:
            modelpath: Path to SDXL model (for tokenizer alignment)
            llmname: HuggingFace model name for LLM
        """
        super().__init__(modelpath)
        self.llmnm = llmname
        self.llmmdl = None
        self.llmtknzr = None

        # Initialize SDXL tokenizer for alignment
        from transformers import AutoTokenizer as SDXLTokenizer
        self.sdxltknzr = SDXLTokenizer.from_pretrained(
            modelpath if modelpath else "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder='tokenizer',
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # Lightweight SpaCy for tokenization only (not dependency parsing)
        # we tried using the transformer model but it was way too slow
        self.spcynlp = spacy.load("en_core_web_sm")

        print(f"LLM Parser initialized with {llmname}")

    def loadllm(self):
        """Load LLM model (sequential approach - load before parsing)."""
        if self.llmmdl is None:
            print(f"Loading {self.llmnm}...")
            print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

            # 4-bit quantization keeps it around 5-6GB which is nice
            # tried 8-bit but kept running out of VRAM on my 3090
            self.llmmdl = AutoModelForCausalLM.from_pretrained(
                self.llmnm,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_4bit=True,
            )
            self.llmtknzr = AutoTokenizer.from_pretrained(self.llmnm)

            print(f"LLM loaded! GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    def unloadllm(self):
        """Unload LLM to free GPU memory for SDXL."""
        if self.llmmdl is not None:
            print("Unloading LLM to free memory for SDXL...")
            import gc  # lazy import, sue me

            del self.llmmdl
            del self.llmtknzr
            self.llmmdl = None
            self.llmtknzr = None

            gc.collect()
            torch.cuda.empty_cache()
            # sometimes you gotta call it twice, idk why but it helps
            torch.cuda.empty_cache()

            print(f"LLM unloaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    def llmsemanticparsing(self, prmpt: str) -> Dict[str, Any]:
        """
        Use LLM to extract semantic groups of prompt.

        Args:
            prmpt: Input textprompt

        Returns:
            Dictionary with semantic groups
        """
        # spent way too long tweaking this prompt tbh
        systmprmpt = """You are a semantic parser for image generation prompts.
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
1. Each group has amain object (noun)
2. Attributes are words that describe ONLY that object
3. Use exact words from the input prompt 
4. Include determiners (a, the) in phrase
5. For compound attributes, list each word separately in attribute_words

Examples:

Inputtext: "a red ball and a blue cube"
Output json :
{
  "groups": [
    {"object": "ball", "attributes": ["red"], "phrase": "a red ball", "object_word": "ball", "attribute_words": ["red"]},
    {"object": "cube", "attributes": ["blue"], "phrase": "a blue cube", "object_word": "cube", "attribute_words": ["blue"]}
  ]
}
"""

        mssgs = [
            {"role": "system", "content": systmprmpt},
            {"role": "user", "content": f'Input: "{prmpt}"'}
        ]

        txt = self.llmtknzr.apply_chat_template(
            mssgs,
            tokenize=False,
            add_generation_prompt=True
        )
        npts = self.llmtknzr(txt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            tpts = self.llmmdl.generate(
                **npts,
                max_new_tokens=512,  # usually enough, increase if you get truncated outputs
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.llmtknzr.eos_token_id
            )

        rspns = self.llmtknzr.decode(tpts[0], skip_special_tokens=True)

        # Extract only the assistant's response (after "assistant" marker)
        # this is janky but qwen formats it weird
        if "assistant" in rspns:
            rspns = rspns.split("assistant")[-1].strip()

        # ok so extracting json from llm output is surprisingly annoying
        # tried a bunch of regex patterns, this one works most of the time
        jsnmth = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', rspns, re.DOTALL)
        if jsnmth:
            jsnstr = jsnmth.group()
            try:
                prsd = json.loads(jsnstr)
                return prsd
            except json.JSONDecodeError as e:
                # fallback: sometimes the model outputs extra stuff after the json
                # so try to find just the groups part
                grmth = re.search(r'\{\s*"groups"\s*:\s*\[.*?\]\s*\}', rspns, re.DOTALL)
                if grmth:
                    try:
                        prsd = json.loads(grmth.group())
                        return prsd
                    except json.JSONDecodeError:
                        pass  # give up and fall through to error

                print(f"JSONparse error: {e}")
                print(f"Extracted JSONtext: {jsnstr[:200]}...")
                print(f"LLM reponse: {rspns[:500]}...")
                raise ValueError(f"LLM returned invalid JSON: {e}")
        else:
            # this shouldn't really happen but just in case
            print(f"Full LLM response: {rspns}")
            raise ValueError(f"No JSON found in LLM response")

    def findwordsdxltkns(self, wrd: str, sdxltknstrs: List[str]) -> List[int]:
        """
        Find where a word appears in SDXL tokenization.
        Handles multi-wordpiece tokens (e.g., "sunglasses" -> ["sun", "##glasses"]).
        Also handles multi-word phrases (e.g., "stop sign" -> ["stop</w>", "sign</w>"]).

        Args:
            wrd: Word or phrase to find
            sdxltknstrs: SDXL tokenized string

        Returns:
            List of token pos
        """
        wlwr = wrd.lower()
        
        # Handle multi-word phrases (e.g., "stop sign" -> ["stop", "sign"])
        wrdprts = wlwr.split()
        
        # multi-word case - gotta find consecutive tokens
        if len(wrdprts) > 1:
            for strtidx in range(len(sdxltknstrs) - len(wrdprts) + 1):
                mtchpsns = []
                mtchd = True
                
                for i, prt in enumerate(wrdprts):
                    tknidx = strtidx + i
                    if tknidx >= len(sdxltknstrs):
                        mtchd = False
                        break
                    
                    tkn = sdxltknstrs[tknidx]
                    # strip out the special markers
                    tkncln = tkn.lower().replace('</w>', '').replace('##', '')
                    
                    if tkncln == prt:
                        mtchpsns.append(tknidx)
                    else:
                        mtchd = False
                        break
                
                if mtchd and len(mtchpsns) == len(wrdprts):
                    return mtchpsns
        
        # simple case: single word exact match
        for idx, tkn in enumerate(sdxltknstrs):
            tkncln = tkn.lower().replace('</w>', '').replace('##', '')
            if tkncln == wlwr:
                return [idx]

        # ugh ok the word got split across multiple tokens
        # this happens with compound words like "sunglasses" -> "sun" + "glasses"
        for strtidx in range(len(sdxltknstrs)):
            rcnstrctd = ""
            pstns = []

            # look ahead up to 5 tokens (should be enough for most words)
            for ffst in range(min(5, len(sdxltknstrs) - strtidx)):
                tkn = sdxltknstrs[strtidx + ffst]
                tkncln = tkn.lower().replace('</w>', '').replace('##', '')
                rcnstrctd += tkncln
                pstns.append(strtidx + ffst)

                if rcnstrctd == wlwr:
                    return pstns

        # couldn't find it anywhere :(
        print(f"Warning: Could not find '{wrd}' in SDXL tokens: {sdxltknstrs}")
        return []

    def alignsemanticgrps(self, smntcgrps: Dict[str, Any], prmpt: str) -> List[List[List[int]]]:
        """
        Align LLM's semantic groups with SDXL token positions.

        Args:
            smntcgrps: Output from LLM
            prmpt: Original prompt

        Returns:
            token_indices in ToMe format: [[[noun_pos], [attr_pos]], ...]
        """
        # Get SDXL tokenization
        sdxltkns = self.sdxltknzr(prmpt)
        sdxltknstrs = self.sdxltknzr.convert_ids_to_tokens(sdxltkns["input_ids"])

        print(f"SDXL tokens: {sdxltknstrs}")

        llndxs = []

        for grp in smntcgrps["groups"]:
            bjwrd = grp.get("objword") or grp.get("object_word", "")
            ttrbwrds = grp.get("attribwords") or grp.get("attribute_words", [])

            # Find object position in SDXL tokens
            bjctpsns = self.findwordsdxltkns(bjwrd, sdxltknstrs)

            # Find attribute positions
            ttrbtpsns = []
            for ttrwrd in ttrbwrds:
                pstns = self.findwordsdxltkns(ttrwrd, sdxltknstrs)
                ttrbtpsns.extend(pstns)

            if bjctpsns:
                if ttrbtpsns:
                    llndxs.append([bjctpsns, ttrbtpsns])
                    print(f"Aligned: {bjwrd}@{bjctpsns} + {ttrbwrds}@{ttrbtpsns}")
                else:
                    # no attributes, still include the object tho
                    llndxs.append([bjctpsns, []])
                    print(f"Aligned: {bjwrd}@{bjctpsns} (no attributes)")
            else:
                # this usually means the LLM hallucinated a word that's not in the prompt
                print(f"Warning: Could not align group: {grp}")

        return llndxs

    def parse(self, prmpt: str) -> Tuple[List[List[List[int]]], List[str]]:
        """
        Parse prompt using LLM semantic understanding.

        Sequential approach:
        1. Load LLM
        2. Parse prompt
        3. Unload LLM (free memory for SDXL)
        4. Return results

        Args:
            prmpt: Input text prompt

        Returns:
            Tuple of (token_indices, prompt_anchor)
        """
        print(f"\n{'='*60}")
        print(f"LLM Parsing: '{prmpt}'")
        print(f"{'='*60}")

        self.loadllm()

        try:
            # Get semantic groups from LLM
            smntcgrps = self.llmsemanticparsing(prmpt)
            print(f"\nLLM identified {len(smntcgrps['groups'])} semantic groups:")
            for i, grp in enumerate(smntcgrps['groups']):
                print(f"  {i+1}. {grp['phrase']}")

            # Align with SDXL tokenization
            tknndcs = self.alignsemanticgrps(smntcgrps, prmpt)

            # Extract anchor prompts
            pnchr = [grp["phrase"] for grp in smntcgrps["groups"]]

            print(f"\nFinal output:")
            print(f"  token_indices: {tknndcs}")
            print(f"  prompt_anchor: {pnchr}")
            print(f"{'='*60}\n")

            return tknndcs, pnchr

        finally:
            # Always unload LLM even if something explodes
            self.unloadllm()


    def cleanup(self):
        """Clean up LLM resources."""
        self.unloadllm()


# quick test if running directly
if __name__ == "__main__":
    # just for testing, delete later maybe
    parser = LLMParser()
    result = parser.parse("a red cat and a blue dog wearing sunglasses")
    print(result)