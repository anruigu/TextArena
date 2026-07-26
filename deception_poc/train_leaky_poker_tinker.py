#!/usr/bin/env python3
"""Standalone GRPO training for the leaky-lies poker policy on the TINKER backend.

Why Tinker here: the SkyRL path runs the env INSIDE the trainer (needs a local vLLM engine, the
qwen3_5_moe GDN-MoE deps, hydra config, an isolated uv env). Tinker inverts that -- the model +
optimizer live on Tinker (LoRA), sampling is an API call, and OUR process runs the rollout loop.
So the "custom thing" is just: reuse the SAME plain-Python pieces (LeakyPokerEnv + the fixed
behavioral exploiter + LeakReader + the dense leaky_reward) and swap the policy from local vLLM to
Tinker sampling. No vLLM, no hydra, no BaseTextEnv-in-trainer -- it sidesteps the wall we hit.

Neat consequence: the policy runs on Tinker, so the LOCAL GPU is free -- we run the 8B leak-reader
in-process (reader_mode=local), no reader service required.

  # sparse baseline arm (no reader):
  cd /workspace/allie/TextArena/deception_poc
  set -a; . /workspace/allie/.env; set +a
  /workspace/allie/performative/.venv/bin/python train_leaky_poker_tinker.py \
      --reward-mode sparse --opponent-mode exploiter --steps 100

  # dense feature arm (leak-reader on the local GPU):
  CUDA_VISIBLE_DEVICES=0 /workspace/allie/performative/.venv/bin/python train_leaky_poker_tinker.py \
      --reward-mode dense --opponent-mode exploiter --leak-lambda 100 --steps 100

  # offline logic check (no Tinker, scripted policy):
  /workspace/allie/performative/.venv/bin/python train_leaky_poker_tinker.py --dry-run
"""
from __future__ import annotations
import argparse, asyncio, os, random, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/workspace/allie/skyrl-neg-wt/skyrl-gym")
sys.path.insert(0, "/workspace/allie/TextArena")
sys.path.insert(0, "/workspace/allie/performative/scripts")

from skyrl_gym.envs.leaky_poker.env import LeakyPokerEnv, POKER_SYSTEM, ACT_RE  # noqa: E402

D = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- rollout
def build_env(seed, hero, args, shared_reader):
    reader_mode = "none"
    if args.reward_mode == "dense":
        reader_mode = "stub" if args.dry_run else "local"
    env = LeakyPokerEnv(
        env_config={"opponent_mode": args.opponent_mode, "reward_mode": args.reward_mode,
                    "leak_lambda": args.leak_lambda, "reader_mode": reader_mode,
                    "num_rounds": args.num_rounds, "hold_lie_rate": True},
        extras={"reward_spec": {"ground_truth": {"seed": seed, "num_rounds": args.num_rounds,
                                                 "hero": hero}}, "max_turns": args.num_rounds * 12},
    )
    if shared_reader is not None and reader_mode == "local":
        env._local_reader = shared_reader  # share ONE 8B reader across all envs
    return env


async def rollout(seed, hero, args, tok, sample_fn, shared_reader):
    """One heads-up match. Returns dict with token stream (prompt/response/logprobs/mask), reward, metrics."""
    env = build_env(seed, hero, args, shared_reader)
    prompt, _ = env.init([{"role": "system", "content": POKER_SYSTEM}])
    prompt_ids = tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True,
                                         enable_thinking=False)
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    prompt_ids = [int(t) for t in prompt_ids]
    ctx = list(prompt_ids)
    resp_ids, logprobs, mask = [], [], []
    reward_sum, done, steps = 0.0, False, 0
    while not done and steps < args.max_turns:
        steps += 1
        act_ids, act_lp = await sample_fn(ctx)
        if not act_ids:
            break
        text = tok.decode(act_ids, skip_special_tokens=True)
        m = ACT_RE.search(text)
        action = (m.group(0) if m else text.strip()[:12]) or "[Check]"
        # accumulate the sampled action as trainable tokens
        ctx.extend(act_ids); resp_ids.extend(act_ids)
        logprobs.extend(act_lp if act_lp else [0.0] * len(act_ids)); mask.extend([1] * len(act_ids))
        out = env.step(action)
        reward_sum += out["reward"]; done = out["done"]
        if not done and out["observations"]:
            obs_ids = tok.encode(out["observations"][0]["content"], add_special_tokens=False)
            ctx.extend(obs_ids); resp_ids.extend(obs_ids)
            logprobs.extend([0.0] * len(obs_ids)); mask.extend([0] * len(obs_ids))
    env.close()
    return {"prompt_ids": prompt_ids, "response_ids": resp_ids, "logprobs": logprobs,
            "loss_mask": mask, "reward": reward_sum, "metrics": env.get_metrics(),
            "seed": seed, "n_tokens": len(resp_ids)}


def grpo_advantages(rewards, group_size, normalize=True):
    adv = np.zeros(len(rewards), dtype=np.float32)
    for g in range(0, len(rewards), group_size):
        grp = np.array(rewards[g:g + group_size], dtype=np.float32)
        mu = grp.mean()
        sd = grp.std() + 1e-6
        adv[g:g + group_size] = (grp - mu) / sd if normalize else (grp - mu)
    return adv.tolist()


def build_datums(rollouts, advantages, tok, max_len, types, TensorData):
    import torch
    datums = []
    for r, a in zip(rollouts, advantages):
        full = r["prompt_ids"] + r["response_ids"]
        plen = len(r["prompt_ids"])
        if len(full) > max_len:
            full = full[:max_len]
            r_len = len(full) - plen
            lp = r["logprobs"][:r_len]; msk = r["loss_mask"][:r_len]
        else:
            lp = r["logprobs"]; msk = r["loss_mask"]
        if len(full) < 2 or plen >= len(full):
            continue
        target = full[1:]
        full_lp = ([0.0] * plen + list(lp))[1:]
        full_msk = ([0] * plen + list(msk))[1:]
        n = len(target)
        full_lp = (full_lp + [0.0] * n)[:n]
        full_msk = (full_msk + [0] * n)[:n]
        adv = torch.zeros(len(full))
        for i in range(plen, len(full)):
            if i - 1 < len(full_msk) and full_msk[i - 1] > 0:
                adv[i] = a
        adv = adv[1:]
        datums.append(types.Datum(
            model_input=types.ModelInput.from_ints(tokens=full[:-1]),
            loss_fn_inputs={"target_tokens": TensorData.from_torch(torch.tensor(target)),
                            "logprobs": TensorData.from_torch(torch.tensor(full_lp, dtype=torch.float32)),
                            "advantages": TensorData.from_torch(adv.float())}))
    return datums


# --------------------------------------------------------------------------- main
async def main_async(args):
    # shared local leak-reader (dense arm) — policy is on Tinker, so the local GPU is free
    shared_reader = None
    if args.reward_mode == "dense" and not args.dry_run:
        from leaky_reward import LeakReader
        shared_reader = LeakReader(device=args.reader_device, batch=16)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    seeds = [args.seed + i for i in range(args.n_prompts_total)]

    if args.dry_run:
        # scripted sampler: emit a random bracket action; validates the whole pipeline (the env's
        # validity gate/parse handles illegal-in-context actions).
        async def sample_fn(ctx):
            a = random.choice(["[Bet 40]", "[Check]", "[Call]", "[Fold]", "[Raise 40]", "[Bet 120]"])
            ids = tok.encode(a, add_special_tokens=False)
            return ids, [0.0] * len(ids)
        # one small batch, no training
        rolls = await asyncio.gather(*[rollout(s, i % 2, args, tok, sample_fn, shared_reader)
                                       for i, s in enumerate(seeds[:args.batch_prompts * args.group])])
        rewards = [r["reward"] for r in rolls]
        adv = grpo_advantages(rewards, args.group)
        import tinker
        from tinker import types
        from tinker.types.tensor_data import TensorData
        datums = build_datums(rolls, adv, tok, args.max_seq_len, types, TensorData)
        print(f"[dry-run] {len(rolls)} rollouts | reward mean={np.mean(rewards):.2f} "
              f"| mean_bluff_leak={np.mean([r['metrics']['mean_bluff_leakage'] for r in rolls]):.3f} "
              f"| lie_rate={np.mean([r['metrics']['lie_rate'] for r in rolls]):.3f} "
              f"| built {len(datums)} datums | adv[0:4]={[round(x,2) for x in adv[:4]]}")
        print("[dry-run] pipeline OK (env rollout -> reward -> GRPO advantage -> Tinker datum).")
        return

    # ---- real Tinker training ----
    import tinker
    from tinker import types
    from tinker.types.tensor_data import TensorData
    try:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name or
                   f"leakypoker_tinker_{args.opponent_mode}_{args.reward_mode}_lam{args.leak_lambda}",
                   config=vars(args))
    except Exception as e:  # noqa: BLE001 — wandb is optional
        print(f"[tinker] wandb disabled ({e})", flush=True)
        wandb = None

    service = tinker.ServiceClient()
    training = await service.create_lora_training_client_async(base_model=args.model, rank=args.lora_rank)
    print(f"[tinker] LoRA rank {args.lora_rank} on {args.model}", flush=True)

    sem = asyncio.Semaphore(args.max_concurrent)
    rng = random.Random(args.seed)

    for step in range(args.steps):
        t0 = time.time()
        sampling = await training.save_weights_and_get_sampling_client_async(name=f"step{step}")

        async def sample_fn(ctx):
            async with sem:
                res = await sampling.sample_async(
                    prompt=types.ModelInput.from_ints(tokens=ctx), num_samples=1,
                    sampling_params=types.SamplingParams(max_tokens=args.max_gen, temperature=args.temperature, top_p=0.95))
            seq = res.sequences[0]
            return list(seq.tokens), (list(seq.logprobs) if seq.logprobs else [])

        # batch of prompts (seeds) x group samples each
        batch_seeds = [rng.choice(seeds) for _ in range(args.batch_prompts)]
        tasks = []
        for si, s in enumerate(batch_seeds):
            for g in range(args.group):
                tasks.append(rollout(s, (si + g) % 2, args, tok, sample_fn, shared_reader))
        rolls = await asyncio.gather(*tasks)

        rewards = [r["reward"] for r in rolls]
        adv = grpo_advantages(rewards, args.group)
        datums = build_datums(rolls, adv, tok, args.max_seq_len, types, TensorData)
        if not datums:
            print(f"[step {step}] no datums, skipping", flush=True); continue

        fb = training.forward_backward(datums, loss_fn=args.loss_fn)
        opt = training.optim_step(types.AdamParams(learning_rate=args.lr))
        fb.result(); opt.result()

        m = {
            "reward/mean": float(np.mean(rewards)),
            "reward/std": float(np.std(rewards)),
            "bluff/mean_leakage": float(np.mean([r["metrics"]["mean_bluff_leakage"] for r in rolls])),
            "bluff/fold_rate": float(np.mean([r["metrics"]["bluff_fold_rate"] for r in rolls])),
            "bluff/lie_rate": float(np.mean([r["metrics"]["lie_rate"] for r in rolls])),
            "hero/net_chips": float(np.mean([r["metrics"]["hero_net_chips"] for r in rolls])),
            "rollout/mean_tokens": float(np.mean([r["n_tokens"] for r in rolls])),
            "time/step_s": time.time() - t0,
        }
        if wandb is not None:
            wandb.log(m, step=step)
        print(f"[step {step:3d}] reward={m['reward/mean']:+.2f} leak={m['bluff/mean_leakage']:.3f} "
              f"fold={m['bluff/fold_rate']:.2f} lie={m['bluff/lie_rate']:.2f} "
              f"net={m['hero/net_chips']:+.0f} ({m['time/step_s']:.0f}s)", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0:
            try:
                p = training.save_state(name=f"state_{step+1:05d}").result().path
                print(f"  saved state -> {p}", flush=True)
            except Exception as e:
                print(f"  save_state failed: {e}", flush=True)
    if wandb is not None:
        wandb.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--reward-mode", default="dense", choices=["dense", "sparse"])
    ap.add_argument("--opponent-mode", default="exploiter", choices=["exploiter", "scripted"])
    ap.add_argument("--leak-lambda", type=float, default=100.0)
    ap.add_argument("--num-rounds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-prompts", type=int, default=8, help="distinct seeds per step")
    ap.add_argument("--group", type=int, default=8, help="samples per seed (GRPO group)")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--loss-fn", default="importance_sampling")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-gen", type=int, default=16, help="tokens per action (a bracket token is tiny)")
    ap.add_argument("--max-turns", type=int, default=48)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--max-concurrent", type=int, default=32)
    ap.add_argument("--n-prompts-total", type=int, default=2048)
    ap.add_argument("--reader-device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--wandb-project", default="leaky-poker-tinker")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
