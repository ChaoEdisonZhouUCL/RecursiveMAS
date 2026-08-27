"""CPU checks for training-time --self_inject, using the real module functions."""
import sys
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import train_outerlinks_math500 as T
from prompts import SELF_INJECT_LABEL

D = 16
VOCAB = 100


class Enc(dict):
    def to(self, *a, **k):
        return self


class FakeTok:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        n = max(1, len(text.split()))
        ids = torch.arange(2, 2 + n).unsqueeze(0) % VOCAB
        return Enc(input_ids=ids, attention_mask=torch.ones_like(ids))


EMB = torch.nn.Embedding(VOCAB, D)


def embed_fn(ids):
    return EMB(ids)


def n_label_tokens(role):
    return FakeTok()(SELF_INJECT_LABEL[role])["input_ids"].size(1)


fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


# ── 1. labels are role-distinct and shared with inference ────────────────────
print("\n[1] label table")
vals = list(SELF_INJECT_LABEL.values())
check(len(set(vals)) == 3, "three distinct role labels: %r" % (vals,))
import inference_utils.inference_mas as IM
check(IM.SELF_INJECT_LABEL is SELF_INJECT_LABEL,
      "inference imports the SAME dict object (cannot drift)")


# ── 2. layout: [pre][latent][label+block][post] ──────────────────────────────
print("\n[2] embed layout")
B, L_lat, L_si = 2, 5, 4
pre = ["hello world foo"] * B
post = ["tail text"] * B
latent = torch.randn(B, L_lat, D)
block = torch.randn(B, L_si, D)

ie0, am0 = T._build_input_embeds_batch(FakeTok(), embed_fn, pre, post, latent,
                                       torch.device("cpu"), torch.float32)
ie1, am1 = T._build_input_embeds_batch(FakeTok(), embed_fn, pre, post, latent,
                                       torch.device("cpu"), torch.float32,
                                       self_inject=block)
check(ie1.size(1) == ie0.size(1) + L_si,
      "injected seq is exactly L_si longer (%d vs %d)" % (ie1.size(1), ie0.size(1)))
check(am1.sum().item() == am0.sum().item() + B * L_si,
      "attention mask covers the injected positions")
# the injected block must sit immediately after the latent, before post
n_pre = FakeTok()(pre[0])["input_ids"].size(1)
got = ie1[0, n_pre + L_lat: n_pre + L_lat + L_si]
check(torch.allclose(got, block[0]), "block occupies the slot right after the latent")
check(torch.allclose(ie0, T._build_input_embeds_batch(
          FakeTok(), embed_fn, pre, post, latent, torch.device("cpu"),
          torch.float32, self_inject=None)[0]),
      "self_inject=None is byte-identical to the control path")


# ── 3. round 0 injects nothing; later rounds do ──────────────────────────────
print("\n[3] round schedule")
T.SELF_INJECT.reset()
T.SELF_INJECT.enabled = True
T.SELF_INJECT.keep_grad = False
seen = []
for r in range(3):
    per_round = {}
    for role in ("planner", "critic", "solver"):
        blk = T._self_inject_block(FakeTok(), embed_fn, role, B,
                                   torch.device("cpu"), torch.float32)
        per_round[role] = None if blk is None else tuple(blk.shape)
        T.SELF_INJECT.record(role, torch.randn(B, 7, D))
    T.SELF_INJECT.commit()
    seen.append(per_round)
    print("    round %d: %s" % (r, per_round))

check(all(v is None for v in seen[0].values()), "round 0 injects nothing for any role")
for r in (1, 2):
    for role in ("planner", "critic", "solver"):
        exp = (B, n_label_tokens(role) + 7, D)
        check(seen[r][role] == exp,
              "round %d %s block is label+latents %s" % (r, role, exp))


# ── 4. per-micro-batch reset ─────────────────────────────────────────────────
print("\n[4] micro-batch isolation")
T.SELF_INJECT.reset()
after = T._self_inject_block(FakeTok(), embed_fn, "planner", B,
                            torch.device("cpu"), torch.float32)
check(after is None, "reset() clears the store, so the next rollout opens clean")

# batch-size guard
T.SELF_INJECT.reset()
T.SELF_INJECT.record("planner", torch.randn(B, 7, D))
T.SELF_INJECT.commit()
check(T._self_inject_block(FakeTok(), embed_fn, "planner", B + 1,
                           torch.device("cpu"), torch.float32) is None,
      "mismatched batch size refuses to inject rather than mixing samples")


# ── 5. disabled == control ───────────────────────────────────────────────────
print("\n[5] disabled path")
T.SELF_INJECT.reset()
T.SELF_INJECT.enabled = False
T.SELF_INJECT.record("planner", torch.randn(B, 7, D))
T.SELF_INJECT.commit()
check(T._self_inject_block(FakeTok(), embed_fn, "planner", B,
                           torch.device("cpu"), torch.float32) is None,
      "enabled=False records nothing and injects nothing")


# ── 6. gradient policy ───────────────────────────────────────────────────────
print("\n[6] gradient policy")
for keep in (False, True):
    T.SELF_INJECT.reset()
    T.SELF_INJECT.enabled = True
    T.SELF_INJECT.keep_grad = keep
    src = torch.randn(B, 7, D, requires_grad=True)
    h = src * 2.0                      # stand-in for the agent's rollout
    T.SELF_INJECT.record("planner", h)
    T.SELF_INJECT.commit()
    blk = T._self_inject_block(FakeTok(), embed_fn, "planner", B,
                               torch.device("cpu"), torch.float32)
    if blk.requires_grad:
        blk.sum().backward()
    got = src.grad is not None and src.grad.abs().sum().item() > 0
    check(got == keep,
          "keep_grad=%s -> gradient reaches the previous round: %s" % (keep, got))
    check(blk.requires_grad == keep,
          "keep_grad=%s -> injected block requires_grad=%s" % (keep, blk.requires_grad))

T.SELF_INJECT.reset()
T.SELF_INJECT.enabled = False

print("\n%d failure(s)" % len(fails))
for f in fails:
    print("  - " + f)
sys.exit(1 if fails else 0)
