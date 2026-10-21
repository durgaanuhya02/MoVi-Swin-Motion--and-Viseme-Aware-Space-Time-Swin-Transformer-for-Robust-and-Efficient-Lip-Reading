"""Prompt formats shared by LLM training and inference.

Two tasks are trained into the same LoRA adapter:
    reconstruct:  phonemes -> sentence                         (VALLR's Stage 2)
    repair:       phonemes + draft with [?] words -> sentence  (our Repair step)
"""
MASK = "[?]"


def reconstruct_prompt(phonemes):
    return f"### Phonemes: {' '.join(phonemes)}\n### Sentence:"


def repair_prompt(phonemes, draft_words):
    return f"### Phonemes: {' '.join(phonemes)}\n### Draft: {' '.join(draft_words)}\n### Sentence:"


def target(text):
    return f" {text}\n"
