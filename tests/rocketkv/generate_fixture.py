#!/usr/bin/env python3
"""Generate fixtures by executing the read-only NVlabs reference on CPU."""
# RocketKV research implementation. See docs/rocketkv/NOTICE.
import argparse
from collections import namedtuple
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import torch.nn.functional as F

TopKResult = namedtuple("TopKResult", ("values", "indices"))


def stable_topk(x, k, dim=-1, largest=True, sorted=True, *, out=None):
    if out is not None:
        raise ValueError("out is not supported in the fixture hook")
    ids = torch.argsort(x, dim=dim, descending=largest, stable=True)
    ids = ids.narrow(dim, 0, k)
    return TopKResult(x.gather(dim, ids), ids)


def case(ref, seed, heads, kv_heads, n, d, window, capacity, page, r, count, kernel):
    torch.manual_seed(seed)
    q_obs = torch.randn(1, heads, window, d)
    keys = torch.randn(1, kv_heads, n, d).half().float()
    values = torch.randn_like(keys).half().float()
    query = torch.randn(1, heads, 1, d)
    config = ref.RocketArgs(window_size=(window,), kernel_size=(kernel,))
    cache = ref.RocketKVCache(1, n + 1, capacity, capacity, page, kv_heads, d, config, torch.float32)
    cache.orig_k_cache[:, :, :n] = keys
    cache.orig_v_cache[:, :, :n] = values
    captures = []

    def capture_topk(x, k, dim=-1, largest=True, sorted=True):
        result = stable_topk(x, k, dim, largest, sorted)
        captures.append((x.detach().clone(), result.indices.detach().clone()))
        return result

    # LAST_PREFILL slices [:input_pos[-1]]. A sentinel at n avoids dropping the last prompt key.
    with patch.object(torch.Tensor, "topk", capture_topk):
        cache.update(torch.tensor([n]), q_obs, torch.zeros(1, kv_heads, 1, d),
                     torch.zeros(1, kv_heads, 1, d), ref.LAST_PREFILL)
    indices = torch.cat([captures[0][1].sort().values,
                         torch.arange(n - window, n).expand(1, kv_heads, window)], -1)
    kc = keys.gather(2, indices[..., None].expand(-1, -1, -1, d))
    vc = values.gather(2, indices[..., None].expand(-1, -1, -1, d))
    pad = (-capacity) % page
    kmin = F.pad(kc, (0, 0, 0, pad), value=float("inf")).reshape(1, kv_heads, -1, page, d).amin(-2)
    kmax = F.pad(kc, (0, 0, 0, pad), value=-float("inf")).reshape(1, kv_heads, -1, page, d).amax(-2)
    meta = torch.cat([kmin, kmax], -1) if page > 1 else kc
    mask = torch.ones(1, 1, 1, capacity, dtype=torch.bool)
    captures.clear()
    with patch("torch.topk", capture_topk):
        output = ref.rocket_attn(query, meta, kc, vc, mask, page, r, count, config)
    dims, sparse_ids = captures[0][1], captures[1][1]
    token_scores = captures[1][0]
    qg = query.reshape(1, kv_heads, heads // kv_heads, 1, d)
    qhat = qg.gather(-1, dims.expand(-1, -1, heads // kv_heads, -1, -1))
    signed_ids = torch.where(qhat.sum(2, keepdim=True) > 0, dims + d, dims) if page > 1 else dims
    khat = meta.unsqueeze(2).gather(-1, signed_ids.expand(-1, -1, -1, meta.shape[-2], -1))
    scale = (d * qhat.abs().sum(-1, keepdim=True) / qg.abs().sum(-1, keepdim=True)).sqrt()
    page_scores = ((qhat @ khat.transpose(-1, -2)) / scale).reshape(heads, -1)
    group_keys = keys.repeat_interleave(heads // kv_heads, 1)
    raw = (q_obs @ group_keys.transpose(-1, -2)) / d**0.5
    causal = torch.arange(n)[None, :] <= torch.arange(n - window, n)[:, None]
    probabilities = raw.masked_fill(~causal, -float("inf")).softmax(-1)
    snap_scores = probabilities[..., :n-window].sum(-2).reshape(kv_heads, heads // kv_heads, -1).sum(1)
    gather_ids = sparse_ids[:, :, 0, 0, :, None].expand(-1, -1, -1, d)
    exact_scores = (qg @ kc.unsqueeze(2).transpose(-1, -2) / d**0.5).softmax(-1).sum(2, keepdim=True)
    exact_ids = stable_topk(exact_scores, count).indices
    exact_gather = exact_ids[:, :, 0, 0, :, None].expand(-1, -1, -1, d)
    exact_k = kc.gather(2, exact_gather).repeat_interleave(heads // kv_heads, 1)
    exact_v = vc.gather(2, exact_gather).repeat_interleave(heads // kv_heads, 1)
    exact_output = F.scaled_dot_product_attention(query, exact_k, exact_v)

    def flat(x):
        return x.flatten().tolist()

    return {
        "name": f"seed{seed}_gqa{heads//kv_heads}_page{page}",
        "heads": heads, "kv_heads": kv_heads, "n": n, "d": d, "window": window,
        "capacity": capacity, "page": page, "r": r, "k": count, "kernel": kernel,
        "q_obs": flat(q_obs), "keys": flat(keys), "values": flat(values), "query": flat(query),
        "snap_scores": flat(snap_scores), "pooled": flat(F.max_pool1d(snap_scores, kernel, 1, kernel//2)),
        "retained": flat(indices), "compact_k": flat(kc), "compact_v": flat(vc),
        "kmin": flat(kmin), "kmax": flat(kmax), "dims": flat(dims),
        "page_scores": flat(page_scores), "token_scores": flat(token_scores), "selected": flat(sparse_ids),
        "gather_k": flat(kc.gather(2, gather_ids)), "gather_v": flat(vc.gather(2, gather_ids)),
        "output": flat(output), "exact_selected": flat(exact_ids), "exact_output": flat(exact_output),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("fixture.json"))
    args = parser.parse_args()
    source = args.reference / "gpt-fast" / "rocket.py"
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("rocketkv_official", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the reference at {source}")
    ref = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ref
    spec.loader.exec_module(ref)
    ref.__dict__["default_device"] = "cpu"
    torch.set_num_threads(1)
    cases = [
        case(ref, 1701, 4, 2, 17, 8, 3, 10, 2, 4, 4, 3),
        case(ref, 1702, 4, 1, 13, 8, 2, 7, 3, 3, 3, 1),
        case(ref, 1703, 2, 2, 11, 8, 2, 8, 1, 8, 5, 3),
        case(ref, 1704, 6, 2, 67, 128, 7, 34, 4, 45, 12, 3),
    ]
    payload = {
        "reference_commit": "63637c8eb4b06b5eb83cd687723b0eeb19f48f65",
        "reference": "NVlabs/RocketKV gpt-fast/rocket.py",
        "tie_rule": "descending score, ascending original index; override only torch topk",
        "snap_boundary": "sentinel workaround for official exclusive LAST_PREFILL slice; chronological compaction",
        "dtype": "float32 arithmetic; keys and values rounded to float16",
        "torch_version": torch.__version__,
        "cases": cases,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(cases)} official-reference cases to {args.output}")


if __name__ == "__main__":
    main()
