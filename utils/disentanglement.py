

import torch
from einops import rearrange, repeat, reduce 
import numpy as np


def get_eps(t):
    finfo = torch.finfo(t.dtype)
    baseeps = finfo.eps
    return baseeps * 100  # slightly conservative fudge factor


def flatten(x): # Flatten via einops
    return rearrange(x, "... -> (-1)")

def orthoproj(v, u):
    epsval = get_eps(v)

    vflat = flatten(v)
    uflat = flatten(u)

    dot = torch.vdot(vflat, uflat)
    unormsq = torch.vdot(uflat, uflat)

    if abs(unormsq.item()) < epsval:
        return torch.zeros_like(v)

    coef = dot / (unormsq + epsval)
    return coef * u


def computescale(
    comptokens,
    bscale = 1.0,
    overlapp= 0.3,
    max_scale = 1.5,
):

    if len(comptokens) < 2:
        return bscale

    sim_sum, count = 0.0, 0
    for i in range(len(comptokens)):
        for j in range(i + 1, len(comptokens)):   
            t1, t2 = comptokens[i], comptokens[j]
            epsal = get_eps(t1)

            num1 = torch.linalg.norm(t1) + epsal
            num2 = torch.norm(t2) + epsal
            flatt1 = flatten(t1)
            flatt2 = flatten(t2)

            cos = torch.clamp(torch.vdot(flatt1, flatt2) / (num1 * num2), -1.0, 1.0)
            sim_sum += np.abs(cos.item())
            count += 1

    if count == 0:
        return bscale

    avg = sim_sum / count

    if avg > overlapp:
        boost = (avg - overlapp) / (1.0 - overlapp)
        scaled = bscale + boost * (max_scale - bscale)
        return min(scaled, max_scale)

    return bscale

def gram_schmidt_disentangle(tokens, preserve_magnitude= True):
    if len(tokens) < 2:
        return tokens

    eps = get_eps(tokens[0])
    orig_norms = [torch.linalg.norm(t).item() for t in tokens] if preserve_magnitude else None

    out = []
    for i, vec in enumerate(tokens):
        w = vec.clone()
        for j in range(i):
            proj = orthoproj(w, out[j])
            w = w - proj

        wn = torch.norm(w)
        if wn > eps and preserve_magnitude:
            scale = orig_norms[i] / (wn.item() + 1e-12)
            w = w * scale

        out.append(w)

    return out


# --- disentanglement-
def orthogonal_disentangle_tokens(
    composite_tokens,
    token_indices,
    scale= 1.0,
    use_adaptive_scale= True,
    use_gram_schmidt= True,
):

    if len(composite_tokens) < 2:
        return composite_tokens

    actual_scale = computescale(composite_tokens, scale) if use_adaptive_scale else scale

    if use_gram_schmidt and len(composite_tokens) > 2:
        return gram_schmidt_disentangle(composite_tokens, preserve_magnitude=True)

    result = []
    for i, tok in enumerate(composite_tokens):
        cleaned = tok.clone()
        for j, other in enumerate(composite_tokens):
            if i == j:
                continue
            cleaned = cleaned - actual_scale * orthoproj(tok, other)
        result.append(cleaned)

    return result


def pairwise(
    token_1,
    token_2,
    scale= 1.0,
):
    t1 = token_1.clone()
    t2 = token_2.clone()
    t1_clean = t1 - scale * orthoproj(t1, t2)
    t2_clean = t2 - scale * orthoproj(t2, t1)

    return t1_clean, t2_clean


def apply_dis(
    promps,
    indices_to_alter,
    use_orthogonal= True,
    scale= 1.0,
    use_adaptive_scale= True,
):

    if not use_orthogonal or len(indices_to_alter) < 2:
        return promps

    if promps.dim() == 3:
        # operate only on batch[0] for now — hacky but workable
        seq = promps[0]
        updated = apply_dis_single(seq, indices_to_alter, scale, use_adaptive_scale)
        out = promps.clone()
        out[0] = updated
        return out

    return apply_dis_single(promps, indices_to_alter, scale, use_adaptive_scale)


def apply_dis_single(
    seq_embeds,
    indices_to_alter,
    scale,
    use_adaptive_scale,
):

    tokens, positions = [], []
    for group in indices_to_alter:
        pos = group[0][0]
        positions.append(pos)
        tokens.append(seq_embeds[pos].clone())

    disent = orthogonal_disentangle_tokens(tokens, indices_to_alter, scale, use_adaptive_scale, use_gram_schmidt=True)

    out = seq_embeds.clone()
    for pos, new_tok in zip(positions, disent):
        out[pos] = new_tok
    return out


# metrics

def computescore(t1, t2):
    eps = get_eps(t1)
    n1 = torch.linalg.norm(t1) + eps
    n2 = torch.linalg.norm(t2) + eps
    a = t1 / n1
    b = t2 / n2

    aflat = flatten(a)
    bflat = flatten(b)
    cos = torch.matmul(aflat, bflat)
    return abs(cos.item())


def compute_semantic_overlap(t1, t2):
    t1flat = flatten(t1)
    t2flat = flatten(t2)
    return abs(torch.vdot(t1flat, t2flat).item())


def compute_mutual_orthogonality(tokens):
    l = len(tokens)
    if l < 2:
        return 0
    total=0 
    count = 0
    for i in range(l):
        for j in range(i + 1, l):
            total += computescore(tokens[i], tokens[j])
            count += 1

    return total / count if count else 0



    
