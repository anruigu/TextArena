#!/usr/bin/env python3
"""Re-elicit POSTERIOR value-beliefs from saved TextArena probe games with a large
token budget, fixing the truncation that made reasoning models (qwen3.6, qwen3.5-9b)
fail to emit their <estimate> line at est_max_tokens=2048.

The posterior belief depends only on a seat's end-of-game message history, which the
saved transcript fully determines — so we reconstruct each seat's history and re-ask
the private estimate question at a big budget, WITHOUT replaying the (expensive)
games. Writes `pairs_v2` (+ raw text) back into each game_*.json.

Usage:
  python3 reelicit_ta_posteriors.py --dirs results/vprobe_ta_2p,results/vprobe_ta_3p \
      --est-max-tokens 8192 --concurrency 6
"""
from __future__ import annotations
import argparse, asyncio, json, os, re
from pathlib import Path

SYSTEM = None  # taken from run_crossplay to match exactly
ESTIMATE_RE = re.compile(r"<estimate>\s*(\{.*?\})\s*</estimate>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def load_env_file(path="/workspace/allie/.env"):
    p = Path(path).expanduser()
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = re.sub(r"^export\s+", "", line.strip())
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("\"'")
        if k and v and k not in os.environ:
            os.environ[k] = v


# ---- scoring, copied verbatim from run_value_probe_ta / the DnD probe ----
def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); ranks = [0.0] * len(xs); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a, b):
    n = len(a); ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a); vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return None
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / ((va ** 0.5) * (vb ** 0.5))


def _spearman(a, b):
    return _pearson(_ranks(a), _ranks(b))


def _cosine(a, b):
    na = sum(x * x for x in a) ** 0.5; nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return None
    return sum(a[i] * b[i] for i in range(len(a))) / (na * nb)


def _norm_mae(est, truth):
    se, st = sum(est), sum(truth)
    if se == 0 or st == 0:
        return None
    pe = [x / se for x in est]; pt = [y / st for y in truth]
    return sum(abs(pe[i] - pt[i]) for i in range(len(est))) / len(est)


def _top1(est, truth):
    pred = max(range(len(truth)), key=lambda i: est[i])
    return 1.0 if truth[pred] == max(truth) else 0.0


def score_estimate(est, truth):
    if est is None:
        return {"parsed": 0.0, "spearman": None, "cosine": None, "top1": None, "norm_mae": None}
    return {"parsed": 1.0, "spearman": _spearman(est, truth), "cosine": _cosine(est, truth),
            "top1": _top1(est, truth), "norm_mae": _norm_mae(est, truth)}


def parse_estimate(text, resource_names):
    if not text:
        return None
    text = _THINK_RE.sub("", text)
    m = ESTIMATE_RE.search(text)
    blob = m.group(1) if m else text
    vals = {}
    try:
        d = json.loads(blob)
        if isinstance(d, dict):
            vals = d
    except Exception:  # noqa: BLE001
        vals = {}
    out = []
    for r in resource_names:
        v = vals.get(r)
        if v is None:
            mm = re.search(rf'["\']?{re.escape(r)}["\']?\s*[:=]\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
            v = float(mm.group(1)) if mm else 0.0
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(0.0)
    return out if sum(abs(x) for x in out) else None


def _estimate_request(resource_names, target_id):
    ex = "{" + ", ".join(f'"{r}": 0' for r in resource_names) + "}"
    return (
        f"Pause the negotiation for a private side-question — Player {target_id} (and everyone "
        "else) will NOT see this and it does NOT affect the game. Based on everything you know "
        f"so far, estimate how many points EACH resource is worth to PLAYER {target_id} (their "
        "hidden per-unit values, from THEIR point of view). Give your single best numeric guess "
        "for every resource even if you are unsure. Reply with exactly one line of the form:\n"
        f"<estimate>{ex}</estimate>\n"
        "Use whatever numeric scale you think their values are on. Output nothing but that line."
    )


async def chat(client, model, messages, max_tokens, retries=4):
    kw = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7}
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(**kw)
            if not getattr(resp, "choices", None):
                return ""
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            em = str(e).lower()
            if "temperature" in em and "temperature" in kw:
                kw.pop("temperature")
            elif "max_tokens" in em and "max_tokens" in kw:
                kw["max_completion_tokens"] = kw.pop("max_tokens")
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2.0 * (attempt + 1))
    return ""


def reconstruct_history(game, pid):
    """Rebuild seat pid's end-of-game message history from the transcript, exactly as
    it existed during play (system + alternating user-obs / assistant-action)."""
    hist = [{"role": "system", "content": game["_system"]}]
    for t in game["transcript"]:
        if t["pid"] != pid:
            continue
        hist.append({"role": "user", "content": t["obs"]})
        hist.append({"role": "assistant", "content": t["action"] or "[Broadcast: (pass)]"})
    return hist


async def main_async(args):
    load_env_file()
    from openai import AsyncOpenAI
    # SYSTEM prompt must match the one used at play time.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_crossplay import SYSTEM as PLAY_SYSTEM
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                         api_key=os.environ["OPENROUTER_API_KEY"], timeout=300.0, max_retries=2)
    sem = asyncio.Semaphore(args.concurrency)

    async def elicit(model, hist, rn, target):
        async with sem:
            txt = await chat(client, model, hist + [{"role": "user",
                             "content": _estimate_request(rn, target)}], args.est_max_tokens)
            return parse_estimate(txt, rn), txt

    for d in args.dirs.split(","):
        d = Path(d.strip())
        files = sorted(d.glob("game_*.json"))
        print(f"=== {d} : {len(files)} games ===", flush=True)
        for fp in files:
            g = json.load(open(fp))
            g["_system"] = PLAY_SYSTEM
            rn = g["resource_names"]
            vals = {int(k): v for k, v in g["values"].items()}
            seat_models = g["seat_models"]
            players = g["players"]
            tasks = []
            meta = []
            for pid in range(players):
                hist = reconstruct_history(g, pid)
                for t in range(players):
                    if t == pid:
                        continue
                    tasks.append(elicit(seat_models[pid], hist, rn, t))
                    meta.append((pid, t))
            res = await asyncio.gather(*tasks)
            pairs_v2 = []
            for (pid, t), (est, raw) in zip(meta, res):
                truth = [float(vals[t][r]) for r in rn]
                pairs_v2.append({
                    "reader_pid": pid, "target_pid": t,
                    "reader_model": seat_models[pid], "target_model": seat_models[t],
                    "post_scores": score_estimate(est, truth), "post_est": est,
                    "post_raw": raw, "truth": truth,
                })
            g.pop("_system", None)
            g["pairs_v2"] = pairs_v2
            fp.write_text(json.dumps(g, indent=1))
            parsed = sum(1 for p in pairs_v2 if p["post_scores"]["parsed"]) / max(1, len(pairs_v2))
            print(f"  {fp.name}: reelicited {len(pairs_v2)} pairs, parse={parsed:.0%}", flush=True)
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="results/vprobe_ta_2p,results/vprobe_ta_3p")
    ap.add_argument("--est-max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
