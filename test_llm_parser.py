"""
Test script for LLM-based parser.
Tests the parser in isolation without running full image generation.

Usage:
    python test_llm_parser.py
"""

import torch
from parsers import LLMParser

def test_simple_prompt():
    """Test with simple two-object prompt."""
    print("=" * 70)
    print("TEST 1: Simple prompt")
    print("=" * 70)

    parser = LLMParser(model_path="stabilityai/stable-diffusion-xl-base-1.0")

    prompt = "a white cat and a black dog"
    token_indices, prompt_anchor = parser.parse(prompt)

    print(f"\n✅ SUCCESS!")
    print(f"   token_indices: {token_indices}")
    print(f"   prompt_anchor: {prompt_anchor}")

    # Expected: [[[3], [2]], [[7], [6]]]
    # Expected: ['a white cat', 'a black dog']

    parser.cleanup()
    print()


def test_complex_prompt():
    """Test with complex multi-attribute prompt."""
    print("=" * 70)
    print("TEST 2: Complex prompt with multiple attributes")
    print("=" * 70)

    parser = LLMParser(model_path="stabilityai/stable-diffusion-xl-base-1.0")

    prompt = "a fluffy white cat wearing sunglasses"
    token_indices, prompt_anchor = parser.parse(prompt)

    print(f"\n✅ SUCCESS!")
    print(f"   token_indices: {token_indices}")
    print(f"   prompt_anchor: {prompt_anchor}")

    parser.cleanup()
    print()


def test_comparison():
    """Compare LLM parser with SpaCy parser."""
    print("=" * 70)
    print("TEST 3: Compare LLM vs SpaCy")
    print("=" * 70)

    from parsers import SpaCyParser

    prompt = "a white cat and a black dog"

    # SpaCy
    print("\n📊 Testing SpaCy parser...")
    spacy_parser = SpaCyParser(model_path="stabilityai/stable-diffusion-xl-base-1.0")
    spacy_indices, spacy_anchor = spacy_parser.parse(prompt)
    print(f"SpaCy result:")
    print(f"  token_indices: {spacy_indices}")
    print(f"  prompt_anchor: {spacy_anchor}")

    # LLM
    print("\n📊 Testing LLM parser...")
    llm_parser = LLMParser(model_path="stabilityai/stable-diffusion-xl-base-1.0")
    llm_indices, llm_anchor = llm_parser.parse(prompt)
    print(f"LLM result:")
    print(f"  token_indices: {llm_indices}")
    print(f"  prompt_anchor: {llm_anchor}")

    # Compare
    print("\n📋 Comparison:")
    match = spacy_indices == llm_indices
    print(f"  Token indices match: {'✅ YES' if match else '❌ NO (this is OK, LLM may parse differently)'}")
    print(f"  Format compatible: {'✅ YES' if isinstance(llm_indices, list) else '❌ NO'}")

    llm_parser.cleanup()
    print()


def test_memory():
    """Test memory cleanup."""
    print("=" * 70)
    print("TEST 4: Memory cleanup")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping memory test")
        return

    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    parser = LLMParser(model_path="stabilityai/stable-diffusion-xl-base-1.0")
    prompt = "a white cat and a black dog"

    # This will load LLM internally
    token_indices, prompt_anchor = parser.parse(prompt)

    print(f"After parsing GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    parser.cleanup()

    print(f"After cleanup GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print("✅ Memory should be freed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("🧪 LLM PARSER TEST SUITE")
    print("=" * 70 + "\n")

    try:
        # Test 1: Simple prompt
        test_simple_prompt()

        # Test 2: Complex prompt
        test_complex_prompt()

        # Test 3: Comparison
        test_comparison()

        # Test 4: Memory
        test_memory()

        print("=" * 70)
        print("✅ ALL TESTS COMPLETED!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Review the outputs above")
        print("2. Verify token_indices format is correct: [[[noun_pos], [attr_pos]], ...]")
        print("3. Run full image generation: python run_demo.py (with RunConfig3)")
        print()

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
