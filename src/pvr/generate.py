"""LLM side of Propose-Verify-Repair: generate candidate sentences and score them."""
import torch

from src.llm.prompts import repair_prompt, reconstruct_prompt, target
from src.text.phonemes import normalize_text


@torch.no_grad()
def _generate(model, tok, prompt, n, max_new_tokens):
    enc = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=max(n, 1),
                         num_return_sequences=max(n, 1), do_sample=False, early_stopping=True,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    seen, cands = set(), []
    for t in texts:
        t = normalize_text(t.split("\n")[0])
        if t and t not in seen:
            seen.add(t); cands.append(t)
    return cands


@torch.no_grad()
def llm_logprob(model, tok, prompt, texts):
    """Sum of log P(sentence | prompt) for each text (target tokens + EOS)."""
    device = next(model.parameters()).device
    p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    seqs = [p_ids + tok(target(t), add_special_tokens=False)["input_ids"] + [tok.eos_token_id] for t in texts]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id, device=device)
    att = torch.zeros_like(ids)
    for k, s in enumerate(seqs):
        ids[k, :len(s)] = torch.tensor(s, device=device); att[k, :len(s)] = 1
    logp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
    scores = []
    for k, s in enumerate(seqs):
        tgt = torch.tensor(s[len(p_ids):], device=device)
        pos = torch.arange(len(p_ids) - 1, len(s) - 1, device=device)
        scores.append(logp[k, pos, tgt].sum().item())
    return scores


def propose(model, tok, phonemes, n=5, max_new_tokens=48):
    """1-best (n=1) or N-best candidate sentences with their LLM log-probabilities."""
    prompt = reconstruct_prompt(phonemes)
    cands = _generate(model, tok, prompt, n, max_new_tokens)
    return list(zip(cands, llm_logprob(model, tok, prompt, cands))) if cands else []


def repair(model, tok, phonemes, draft_words, n=5, max_new_tokens=48):
    """Candidates that fill the [?] words of a draft; scored with the plain reconstruct
    prompt so they are comparable with the original candidates."""
    cands = _generate(model, tok, repair_prompt(phonemes, draft_words), n, max_new_tokens)
    return list(zip(cands, llm_logprob(model, tok, reconstruct_prompt(phonemes), cands))) if cands else []
