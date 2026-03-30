#!/usr/bin/env python3
"""
PyTorch Monte-Carlo bilevel weight-learner training for WebShop offline trajectories.

Llama-3.2-3B-Instruct with QLoRA (4-bit quantised) fine-tuning.  Replicates the
four-phase bilevel optimisation loop (update-u, sync u→w, update-w on train+val,
update-φ) from the JAX reference script, but depends only on
PyTorch / HuggingFace / PEFT / bitsandbytes.

Categories are ``goal["product_category"]`` strings.
Simulation eval uses the local WebShop HTTP server
(see ``web_agent_site.webshop_llm_utils``).
"""
from __future__ import annotations

import os
import pickle as pkl
import re
import sys
import time
from collections import deque
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import tyro
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

from llm_rl_scripts.webshop.env.data import (
    build_init_prompt,
    filter_results_by_categories_with_idx,
    get_category_name_from_result,
    load_webshop_offline_json,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_WEBSHOP_ROOT = os.path.join(_REPO_ROOT, "web", "WebShop")
if _WEBSHOP_ROOT not in sys.path:
    sys.path.insert(0, _WEBSHOP_ROOT)


# ---------------------------------------------------------------------------
# System context provided to the model alongside environment observations
# ---------------------------------------------------------------------------

WEBSHOP_SYSTEM = (
    "You are playing Webshop. You will see a text view of the current page "
    "and must output exactly one action per turn.\n"
    "\n"
    "CRITICAL RULES:\n"
    "- search[query] — type a search query. Your query MUST be short "
    "(4 to 8 words maximum). Extract only the core keywords "
    '(e.g., "mens tuxedo shirt").\n'
    "- You only get ONE search. Once you see the search results, you MUST "
    "choose the product with the best price and attributes matching the "
    "instruction and use click[].\n"
    "- Once you see details of an item, you can click on attributes relevant "
    "to instruction and click[Buy Now]\n"
    "- If the attributes or price don't match the instruction and you think "
    "there is a better item, you can click[< Prev] to go back to the "
    "previous page and click on the next item.\n"
    "\n"
    "Actions:\n"
    "- search[query] — search for the core item.\n"
    "- click[button or option] — click a button or option shown in brackets, "
    "e.g. click[Buy Now], click[B078GWRC1J].\n"
    "\n"
    "Output only one action per message, in the form action[argument]. "
    "Nothing else."
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def synth_data_basename(
    K: int, J: int,
    train_task_frac: float, poisoning_frac: float, sample_frac: float,
    val_eval_same_distr: bool, val_split: float, seed: int,
) -> str:
    return (
        f"synth_tf{train_task_frac}_p{poisoning_frac}_sf{sample_frac}"
        f"_ve{val_eval_same_distr}_vs{val_split}_seed{seed}_K{K}_J{J}"
    )


def parse_action_from_llm(text: str) -> Optional[str]:
    text = (text or "").strip()
    for line in text.split("\n"):
        line = line.strip()
        if re.match(r"^(search|click|think)\s*\[", line, re.IGNORECASE):
            m = re.match(r"(search|click|think)\s*\[(.*)\]", line, re.IGNORECASE | re.DOTALL)
            if m:
                kind, arg = m.group(1).lower(), m.group(2).strip()
                return f"{kind}[{arg}]"
            return line
    return None


# ---------------------------------------------------------------------------
# MC data processing  (replaces LLM_RL MCData / TextTrajectory pipeline)
# ---------------------------------------------------------------------------

def _create_mc_sample(
    result: Dict[str, Any],
    tokenizer,
    max_length: int,
    gamma: float,
    system_context: str = "",
) -> Optional[Dict[str, np.ndarray]]:
    """Convert one offline rollout JSON dict into a tokenised MC training sample."""
    goal = result.get("goal") or {}
    instruction = result.get("instruction") or goal.get("instruction_text", "")
    init_prompt = build_init_prompt(instruction)
    log_entries = result.get("log") or []
    terminal_reward = float(result.get("reward", 0.0))

    reset_i = next((k for k, e in enumerate(log_entries) if e.get("action") == "reset"), None)
    if reset_i is None:
        return None

    obs0 = log_entries[reset_i].get("observation", "")

    prefix = (system_context + "\n\n") if system_context else ""

    # (text, is_action, step_reward)
    segments: List[Tuple[str, bool, float]] = []
    segments.append((prefix + init_prompt + obs0 + "\n\nAction:", False, 0.0))

    for j in range(reset_i + 1, len(log_entries)):
        entry = log_entries[j]
        action = entry.get("action", "")
        obs = entry.get("observation", "")
        if action == "reset":
            continue
        is_terminal = bool(entry.get("done", False))
        r = terminal_reward if is_terminal else 0.0
        segments.append((" " + action + "\n", True, r))
        segments.append(("Observation: " + obs + "\n\nAction:", False, 0.0))

    all_ids: List[int] = []
    all_is_action: List[bool] = []
    all_rewards: List[float] = []

    for text, is_action, reward in segments:
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        all_is_action.extend([is_action] * len(ids))
        seg_rewards = [0.0] * len(ids)
        if is_action and reward != 0.0 and len(ids) > 0:
            seg_rewards[-1] = reward
        all_rewards.extend(seg_rewards)

    L = len(all_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if L > max_length:
        all_ids = all_ids[:max_length]
        all_is_action = all_is_action[:max_length]
        all_rewards = all_rewards[:max_length]
    elif L < max_length:
        pad_len = max_length - L
        all_ids += [pad_id] * pad_len
        all_is_action += [False] * pad_len
        all_rewards += [0.0] * pad_len

    returns = [0.0] * max_length
    running = 0.0
    for t in reversed(range(max_length)):
        running = all_rewards[t] + gamma * running
        returns[t] = running

    return {
        "input_ids": np.array(all_ids, dtype=np.int64),
        "should_take_action": np.array(all_is_action, dtype=np.bool_),
        "returns": np.array(returns, dtype=np.float32),
    }


class MCDatasetWithCategory(Dataset):
    """Pre-tokenised MC dataset with per-sample category and frozen embedding."""

    def __init__(
        self,
        results: List[Dict[str, Any]],
        categories: List[int],
        tokenizer,
        max_length: int,
        gamma: float,
        embeddings: Optional[np.ndarray] = None,
        system_context: str = "",
    ):
        self.samples: List[Dict[str, np.ndarray]] = []
        self.categories: List[int] = []
        self.embeddings = embeddings
        for idx, (res, cat) in enumerate(zip(results, categories)):
            s = _create_mc_sample(res, tokenizer, max_length, gamma, system_context=system_context)
            if s is None:
                continue
            s["category"] = np.array(cat, dtype=np.int32)
            if embeddings is not None:
                s["embedding"] = embeddings[idx].astype(np.float32)
            self.samples.append(s)
            self.categories.append(cat)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def mc_collate_fn(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0]:
        out[key] = torch.from_numpy(np.stack([b[key] for b in batch]))
    return out


def _infinite_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True):
    while True:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            collate_fn=mc_collate_fn, drop_last=True,
        )
        yield from loader


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class QHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias_init: float = -4.4):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias_init)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)


class WeightPredictorHead(nn.Module):
    """φ: embedding → scalar logit (used as softmax weights over mini-batch)."""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        in_d = input_dim
        for _ in range(depth - 1):
            layers.extend([nn.Linear(in_d, hidden_dim), nn.ReLU()])
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Simple generation-based policy for simulation rollouts
# ---------------------------------------------------------------------------

class SimplePolicy:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        generation_config: GenerationConfig,
        max_input_length: int,
        device: torch.device,
        system_context: str = "",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.max_input_length = max_input_length
        self.device = device
        self.system_context = system_context

    @torch.no_grad()
    def generate(self, prompt_text: str) -> str:
        full_text = ((self.system_context + "\n\n") if self.system_context else "") + prompt_text
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
            padding=False,
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
        )
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.removesuffix("\n") + "\n"


# ---------------------------------------------------------------------------
# Simulation rollout
# ---------------------------------------------------------------------------

def webshop_rollout_mean_reward(
    policy: SimplePolicy,
    webshop_env,
    goals: List[dict],
    goal_indices: List[int],
    n_rollouts: int,
    max_steps: int,
    runs_prefix: str,
    prompt_truncate_chars: int,
) -> Tuple[float, List[float], List[Dict[str, Any]]]:
    rewards: List[float] = []
    interaction_logs: List[Dict[str, Any]] = []
    if not goal_indices:
        return 0.0, rewards, interaction_logs
    for r in range(n_rollouts):
        gidx = goal_indices[r % len(goal_indices)]
        goal = goals[gidx]
        instruction = goal["instruction_text"]
        init_prompt = build_init_prompt(instruction)
        session_id = f"{runs_prefix}_{gidx}_{r}"
        prompt_suffix = ""
        action = "reset"
        final_reward = 0.0
        curr_log: List[Dict[str, Any]] = []
        for step in range(max_steps + 1):
            try:
                observation, reward, done = webshop_env.step(session_id, action)
            except AssertionError:
                observation = f"ERROR: action {action} is invalid here.\n"
                reward = 0.0
                done = False
            if action.startswith("think["):
                observation = "OK"
            final_reward = float(reward)
            curr_log.append(dict(step=step, action=action, observation=observation,
                                 reward=float(reward), done=bool(done)))
            if step > 0:
                prompt_suffix += f" {action}\nObservation: {observation}\n\nAction:"
            else:
                prompt_suffix += f"{observation}\n\nAction:"
            if done or step >= max_steps:
                break
            trunc = prompt_truncate_chars - len(init_prompt)
            full_prompt = init_prompt + prompt_suffix[-trunc:]
            raw_action = policy.generate(full_prompt).strip()
            parsed = parse_action_from_llm(raw_action)
            if parsed is None:
                first = raw_action.split("\n")[0].strip()
                if first.startswith(("search[", "click[", "think[")):
                    parsed = first
                else:
                    parsed = "think[no valid action]"
                    print(f"No valid action found: {raw_action}")
            action = parsed
        rewards.append(final_reward)
        interaction_logs.append(dict(
            rollout_idx=r, goal_idx=gidx, instruction=instruction,
            final_reward=final_reward, log=curr_log,
        ))
    mean_r = float(np.mean(rewards)) if rewards else 0.0
    return mean_r, rewards, interaction_logs


# ---------------------------------------------------------------------------
# MC loss helpers
# ---------------------------------------------------------------------------

def mc_per_sample_loss(
    q: torch.Tensor,
    q_logits: torch.Tensor,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    should_take_action: torch.Tensor,
    returns: torch.Tensor,
    cql_w: float,
) -> torch.Tensor:
    """Per-sample MC + CQL loss.  All inputs are ``[B, L-1]`` (shifted)."""
    mask = should_take_action.float() * attention_mask.float()
    n_i = mask.sum(dim=1).clamp(min=1.0)

    q_l2 = 0.5 * (q - returns.detach()) ** 2
    q_loss = (q_l2 * mask).sum(dim=1) / n_i

    q_cql = F.cross_entropy(
        q_logits.reshape(-1, q_logits.size(-1)),
        token_ids.reshape(-1),
        reduction="none",
    ).reshape(q.shape)
    q_cql_loss = (q_cql * mask).sum(dim=1) / n_i

    return q_loss + cql_w * q_cql_loss


# ---------------------------------------------------------------------------
# φ diagnostic
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_phi_avg_weight_by_category(
    phi_model: WeightPredictorHead,
    train_embeddings: np.ndarray,
    train_categories: List[int],
    partition_display_names: List[str],
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    n = train_embeddings.shape[0]
    logits_list = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        emb = torch.from_numpy(train_embeddings[start:end].astype(np.float32)).to(device)
        logits_list.append(phi_model(emb).cpu())
    logits_all = torch.cat(logits_list, dim=0)
    weights = torch.softmax(logits_all, dim=0).numpy()
    cats_arr = np.array(train_categories, dtype=np.int32)
    avg: Dict[str, float] = {}
    for k, name in enumerate(partition_display_names):
        m = cats_arr == k
        avg[name] = float(np.sum(weights[m])) if m.sum() > 0 else 0.0
    vals = list(avg.values())
    summary = {
        "phi_avg_weight_mean": float(np.mean(vals)) if vals else 0.0,
        "phi_avg_weight_min": float(np.min(vals)) if vals else 0.0,
        "phi_avg_weight_max": float(np.max(vals)) if vals else 0.0,
        "phi_avg_weight_std": float(np.std(vals)) if len(vals) > 1 else 0.0,
    }
    return avg, summary


# ====================================================================
# Main
# ====================================================================

def main(
    train_data_path: str,
    eval_data_path: str,

    /,

    # Model ----------------------------------------------------------
    model_name_or_path: str = "meta-llama/Llama-3.2-3B-Instruct",

    # QLoRA / quantisation -------------------------------------------
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
    bnb_4bit_compute_dtype: str = "float16",

    # LoRA -----------------------------------------------------------
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,v_proj,k_proj,o_proj",

    # WebShop sim ----------------------------------------------------
    webshop_base_url: Optional[str] = None,
    webshop_sim_max_steps: int = 15,
    webshop_prompt_truncate_chars: int = 6000,

    # Experiment -----------------------------------------------------
    exp_name: Optional[str] = None,
    outputs_path: Optional[str] = None,

    # Logging --------------------------------------------------------
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,

    # Bilevel hypers -------------------------------------------------
    val_split: float = 0.8,
    train_task_frac: float = 0.5,
    num_outer_iter: int = 10,
    num_inner_iter: int = 100,
    inner_opt_steps: int = 1,
    phi_update_factor: float = 1.0,
    tau: float = 1.1,
    init_alpha: float = 0.1,

    # Learning rates -------------------------------------------------
    lr: float = 1e-4,
    lr_phi: Optional[float] = None,
    lr_q_u: Optional[float] = None,
    lr_q_w: Optional[float] = None,
    weight_decay: float = 0.001,

    # Data -----------------------------------------------------------
    poisoning_frac: float = 0.0,
    sample_frac: float = 1.0,
    sample_again_frac: float = 1.0,
    val_eval_same_distr: bool = False,
    seed: int = 0,

    # Training -------------------------------------------------------
    train_bsize: int = 4,
    grad_accum_steps: int = 1,
    max_length: int = 1024,
    bf16: bool = True,
    gradient_checkpointing: bool = False,

    # Q-head / loss --------------------------------------------------
    q_head_bias_init: float = -4.4,
    beta: float = 16.0,
    detach_q: bool = False,
    gamma: float = 0.99,
    cql_weight: float = 0.001,

    # Eval -----------------------------------------------------------
    log_every: int = 256,
    eval_every_steps: Optional[int] = 256,
    skip_eval_simulation: bool = False,
    eval_loss_bsize: int = 32,
    eval_loss_batches: int = 10,

    # Generation (rollout policy) ------------------------------------
    policy_n_rollouts: int = 32,
    policy_max_input_length: int = 256,
    policy_max_output_length: int = 128,
    policy_do_sample: bool = True,
    policy_temperature: Optional[float] = None,
    policy_top_p: Optional[float] = None,
    policy_top_k: Optional[int] = None,

    # Checkpointing --------------------------------------------------
    save_dir: Optional[str] = None,
    max_checkpoints: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,

    # Embeddings / synth data ----------------------------------------
    embeddings_path: Optional[str] = None,

    synth_data_path: Optional[str] = None,
    synth_data_dir: Optional[str] = None,
    synth_embeddings_path: Optional[str] = None,
    synth_K: int = 3,
    synth_J: int = 2,

    # Misc -----------------------------------------------------------
    tokenizer_name_or_path: Optional[str] = None,
    do_time: bool = False,
    hf_token: Optional[str] = None,
):
    # ---------------------------------------------------------------
    # Preamble
    # ---------------------------------------------------------------
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    input_args = dict(locals())
    print(input_args)

    if hf_token is not None:
        from huggingface_hub import login
        login(token=hf_token)
        print("Logged in to Hugging Face using token")

    lr_phi = lr_phi if lr_phi is not None else lr
    lr_q_u = lr_q_u if lr_q_u is not None else lr
    lr_q_w = lr_q_w if lr_q_w is not None else lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if load_in_4bit:
        _compute_dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = _compute_dtype_map.get(bnb_4bit_compute_dtype, torch.float16)
    else:
        dtype = torch.bfloat16 if (bf16 and device.type == "cuda") else torch.float32
    is_main_process = True  # single-process for now
    rng = np.random.RandomState(seed)

    if use_wandb and is_main_process:
        import wandb
        wandb.init(project=wandb_project or "webshop-bilevel-pytorch", config=input_args)

    # ---------------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------------
    tok_path = tokenizer_name_or_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, use_fast=True, use_auth_token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"

    # ---------------------------------------------------------------
    # Offline data
    # ---------------------------------------------------------------
    raw_train = load_webshop_offline_json(os.path.expanduser(train_data_path))
    raw_eval = load_webshop_offline_json(os.path.expanduser(eval_data_path))
    print(f"Original Train Dataset size: {len(raw_train)}")
    print(f"Original Eval  Dataset size: {len(raw_eval)}")

    all_cat_names = sorted(
        {get_category_name_from_result(r) for r in raw_train}
        | {get_category_name_from_result(r) for r in raw_eval}
    )
    if not all_cat_names:
        all_cat_names = ["unknown"]
    cat_name_to_gid: Dict[str, int] = {n: i for i, n in enumerate(all_cat_names)}
    num_categories = len(all_cat_names)

    # ---------------------------------------------------------------
    # Category-based train / eval split
    # ---------------------------------------------------------------
    cat_perm = rng.permutation(num_categories)
    num_train_cats = max(1, int(num_categories * train_task_frac))
    train_category_names = [all_cat_names[int(i)] for i in cat_perm[:num_train_cats]]
    eval_category_names = [all_cat_names[int(i)] for i in cat_perm[num_train_cats:]]

    num_eval_cats = len(eval_category_names)
    num_poison_cats = int(num_eval_cats * poisoning_frac) if num_eval_cats > 0 else 0
    if num_poison_cats > 0:
        poison_perm = rng.permutation(num_eval_cats)
        poison_names = [eval_category_names[int(i)] for i in poison_perm[:num_poison_cats]]
        train_category_names = train_category_names + poison_names
        print(f"Poisoning: added {num_poison_cats} eval categories to train: {poison_names}")

    def get_gid(res):
        return cat_name_to_gid[get_category_name_from_result(res)]

    # ---------------------------------------------------------------
    # Frozen embeddings
    # ---------------------------------------------------------------
    assert embeddings_path is not None, "embeddings_path is required (npz with train_embeddings, eval_embeddings)."
    print("Loading frozen embeddings from", embeddings_path)
    loaded = np.load(os.path.expanduser(embeddings_path), allow_pickle=False)
    train_embeddings_full: np.ndarray = loaded["train_embeddings"]
    eval_embeddings_full: np.ndarray = loaded["eval_embeddings"]

    if val_eval_same_distr:
        raw_eval, eval_subsample_idx = filter_results_by_categories_with_idx(raw_train, eval_category_names)
        eval_embeddings = train_embeddings_full[np.array(eval_subsample_idx)]
    else:
        raw_eval, eval_subsample_idx = filter_results_by_categories_with_idx(raw_eval, eval_category_names)
        eval_embeddings = eval_embeddings_full[np.array(eval_subsample_idx)]

    raw_train, train_subsample_idx = filter_results_by_categories_with_idx(raw_train, train_category_names)
    train_embeddings = train_embeddings_full[np.array(train_subsample_idx)]
    print(f"After category filter: train = {len(raw_train)},  eval = {len(raw_eval)}")
    print(f"Train categories: {train_category_names}")
    print(f"Eval  categories: {eval_category_names}")

    if sample_frac < 1.0:
        assert sample_frac > 0.0
        for tag, arr, emb_arr in [("train", raw_train, train_embeddings),
                                   ("eval", raw_eval, eval_embeddings)]:
            n = len(arr)
            keep = max(1, int(n * sample_frac))
            idx = list(rng.permutation(n)[:keep])
            if tag == "train":
                raw_train = [arr[i] for i in idx]
                train_embeddings = emb_arr[np.array(idx)]
            else:
                raw_eval = [arr[i] for i in idx]
                eval_embeddings = emb_arr[np.array(idx)]
        print(f"After sample_frac: train = {len(raw_train)},  eval = {len(raw_eval)}")

    # ---------------------------------------------------------------
    # Val / Train / Eval split for bilevel
    # ---------------------------------------------------------------
    if val_eval_same_distr:
        print("Splitting val from eval (val has same distribution as eval)")
        n_total = len(raw_eval)
        n_val = max(1, int(n_total * val_split))
        perm = rng.permutation(n_total)
        val_indices = list(perm[:n_val])
        eval_indices = list(perm[n_val:])
        raw_val = [raw_eval[i] for i in val_indices]
        val_embeddings = eval_embeddings[np.array(val_indices)]
        raw_eval_split = [raw_eval[i] for i in eval_indices]
        eval_embeddings = eval_embeddings[np.array(eval_indices)]
        raw_train_final = raw_train
    else:
        print("Splitting val from train (val has same distribution as train)")
        n_total = len(raw_train)
        n_val = max(1, int(n_total * val_split))
        perm = rng.permutation(n_total)
        val_indices = list(perm[:n_val])
        train_indices = list(perm[n_val:])
        raw_train_final = [raw_train[i] for i in train_indices]
        raw_val = [raw_train[i] for i in val_indices]
        val_embeddings = train_embeddings[np.array(val_indices)]
        train_embeddings = train_embeddings[np.array(train_indices)]
        raw_eval_split = raw_eval

    print(f"Data split: {len(raw_train_final)} train, {len(raw_val)} val, {len(raw_eval_split)} eval")

    # ---------------------------------------------------------------
    # Optional synthetic data
    # ---------------------------------------------------------------
    raw_synth: List[Dict[str, Any]] = []
    synth_embeddings: Optional[np.ndarray] = None

    if synth_data_path is not None:
        sp = os.path.expanduser(synth_data_path)
        print("Loading synth data from", sp)
        raw_synth = load_webshop_offline_json(sp)
        assert synth_embeddings_path is not None
        ls = np.load(os.path.expanduser(synth_embeddings_path), allow_pickle=False)
        synth_embeddings = ls.get("synth_embeddings", ls.get("embeddings", None))
        assert synth_embeddings is not None and len(raw_synth) == synth_embeddings.shape[0]
    elif synth_data_dir is not None:
        sb = synth_data_basename(synth_K, synth_J, train_task_frac, poisoning_frac,
                                 sample_frac, val_eval_same_distr, val_split, seed)
        sp = os.path.join(os.path.expanduser(synth_data_dir), sb + ".json")
        if os.path.isfile(sp):
            print("Loading synth data from", sp)
            raw_synth = load_webshop_offline_json(sp)
            assert synth_embeddings_path is not None
            ls = np.load(os.path.expanduser(synth_embeddings_path), allow_pickle=False)
            synth_embeddings = ls.get("synth_embeddings", ls.get("embeddings", None))
            assert synth_embeddings is not None and len(raw_synth) == synth_embeddings.shape[0]
        else:
            print("Synth path not found:", sp, "- proceeding without synth data.")

    if sample_again_frac < 1.0:
        assert 0.0 < sample_again_frac <= 1.0
        if len(raw_synth) > 0 and synth_embeddings is not None:
            assert len(raw_train_final) == len(raw_synth)
            n = len(raw_train_final)
            nk = max(1, int(n * sample_again_frac))
            ki = list(rng.permutation(n)[:nk])
            raw_train_final = [raw_train_final[i] for i in ki]
            train_embeddings = train_embeddings[np.array(ki)]
            raw_synth = [raw_synth[i] for i in ki]
            synth_embeddings = synth_embeddings[np.array(ki)]
            print(f"After sample_again_frac (paired train+synth): kept {nk} of {n}")
        else:
            n = len(raw_train_final)
            nk = max(1, int(n * sample_again_frac))
            ki = list(rng.permutation(n)[:nk])
            raw_train_final = [raw_train_final[i] for i in ki]
            train_embeddings = train_embeddings[np.array(ki)]
            print(f"After sample_again_frac (train only): kept {nk} of {n}")

    # ---------------------------------------------------------------
    # Category mapping for bilevel partitions
    # ---------------------------------------------------------------
    train_categories_raw = [get_gid(c) for c in raw_train_final]
    val_categories_raw = [get_gid(c) for c in raw_val]
    unique_train_categories = sorted(set(train_categories_raw))
    old_to_new = {old: new for new, old in enumerate(unique_train_categories)}
    num_partitions = len(unique_train_categories)
    train_categories = [old_to_new[c] for c in train_categories_raw]
    val_categories = [old_to_new.get(c, 0) for c in val_categories_raw]
    partition_display_names = [all_cat_names[g] for g in unique_train_categories]
    print(f"Bilevel: num_partitions={num_partitions}")
    print(f"Category mapping: {[(partition_display_names[i], i) for i in range(num_partitions)]}")

    if len(raw_synth) > 0 and synth_embeddings is not None:
        synth_categories = [num_partitions + old_to_new.get(get_gid(c), 0) for c in raw_synth]
        raw_train_final = raw_train_final + raw_synth
        train_embeddings = np.concatenate([train_embeddings, synth_embeddings], axis=0)
        train_categories = train_categories + synth_categories
        real_names = list(partition_display_names)
        partition_display_names = real_names + ["synth-" + c for c in real_names]
        num_partitions *= 2
        print(f"Appended {len(raw_synth)} synth samples. num_partitions={num_partitions}")

    # ---------------------------------------------------------------
    # Build MC datasets
    # ---------------------------------------------------------------
    print("Building MC datasets …")
    train_ds = MCDatasetWithCategory(
        raw_train_final, train_categories, tokenizer, max_length, gamma,
        embeddings=train_embeddings, system_context=WEBSHOP_SYSTEM,
    )
    val_ds = MCDatasetWithCategory(
        raw_val, val_categories, tokenizer, max_length, gamma,
        embeddings=val_embeddings, system_context=WEBSHOP_SYSTEM,
    )
    eval_ds = MCDatasetWithCategory(
        raw_eval_split,
        [old_to_new.get(get_gid(r), 0) for r in raw_eval_split],
        tokenizer, max_length, gamma, system_context=WEBSHOP_SYSTEM,
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Eval samples: {len(eval_ds)}")

    train_iter = _infinite_dataloader(train_ds, train_bsize)
    val_iter = _infinite_dataloader(val_ds, train_bsize)

    # ---------------------------------------------------------------
    # Load base model with QLoRA (4-bit quantisation)
    # ---------------------------------------------------------------
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
        print(f"QLoRA: loading {model_name_or_path} in 4-bit ({bnb_4bit_quant_type}, "
              f"double_quant={bnb_4bit_use_double_quant}, compute_dtype={bnb_4bit_compute_dtype})")

    print(f"Loading model {model_name_or_path} …")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map="auto" if load_in_4bit else None,
        trust_remote_code=True,
        use_auth_token=hf_token,
    )

    if load_in_4bit:
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=gradient_checkpointing,
        )
    elif gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[m.strip() for m in lora_target_modules.split(",")],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_cfg)
    if not load_in_4bit:
        model.to(device)
    model.print_trainable_parameters()

    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size

    # ---------------------------------------------------------------
    # Q-heads (u and w) + φ (weight predictor)
    # ---------------------------------------------------------------
    q_head_u = QHead(hidden_size, vocab_size, bias_init=q_head_bias_init).to(device=device, dtype=dtype)
    q_head_w = QHead(hidden_size, vocab_size, bias_init=q_head_bias_init).to(device=device, dtype=dtype)

    embedding_dim = train_embeddings.shape[1] if train_embeddings is not None else hidden_size
    phi_hidden = min(256, embedding_dim)
    phi_model = WeightPredictorHead(embedding_dim, phi_hidden, depth=2).to(device)

    # ---------------------------------------------------------------
    # Optimizers
    # ---------------------------------------------------------------
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt_base = torch.optim.AdamW(lora_params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay)
    opt_u = torch.optim.AdamW(q_head_u.parameters(), lr=lr_q_u, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay)
    opt_w = torch.optim.AdamW(q_head_w.parameters(), lr=lr_q_w, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay)
    opt_phi = torch.optim.Adam(phi_model.parameters(), lr=lr_phi)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # ---------------------------------------------------------------
    # Resume
    # ---------------------------------------------------------------
    loop_state: Dict[str, Any] = {}
    if resume_from_checkpoint is not None:
        rp = os.path.expanduser(resume_from_checkpoint)
        ckpt = torch.load(os.path.join(rp, "checkpoint.pt"), map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        q_head_w.load_state_dict(ckpt["q_head_w"])
        q_head_u.load_state_dict(ckpt.get("q_head_u", ckpt["q_head_w"]))
        phi_model.load_state_dict(ckpt["phi_model"])
        opt_base.load_state_dict(ckpt.get("opt_base", opt_base.state_dict()))
        opt_w.load_state_dict(ckpt.get("opt_w", opt_w.state_dict()))
        opt_phi.load_state_dict(ckpt.get("opt_phi", opt_phi.state_dict()))
        loop_state = ckpt.get("loop_state", {})
        print(f"Resumed from {rp}")

    # ---------------------------------------------------------------
    # Convenience: forward + per-sample loss
    # ---------------------------------------------------------------
    autocast_ctx = partial(torch.cuda.amp.autocast, dtype=dtype, enabled=(dtype != torch.float32))

    def _forward_per_sample(q_head: QHead, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return per-sample MC loss ``[B]``."""
        input_ids = batch["input_ids"].to(device)
        attn_mask = (input_ids != pad_token_id).long()
        sta = batch["should_take_action"].to(device)
        rets = batch["returns"].to(device)

        with autocast_ctx():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            hidden = out.hidden_states[-1]
            if detach_q:
                hidden = hidden.detach()
            q_out = q_head(hidden)

        q_out_f = q_out.float()
        q = torch.gather(q_out_f[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        q_logits = q_out_f[:, :-1]

        return mc_per_sample_loss(
            q, q_logits,
            input_ids[:, 1:],
            attn_mask[:, 1:],
            sta[:, 1:],
            rets[:, 1:],
            cql_weight,
        )

    def _weighted_train_loss(q_head: QHead, batch: Dict[str, torch.Tensor], detach_phi_: bool) -> torch.Tensor:
        per_sample = _forward_per_sample(q_head, batch)
        emb = batch["embedding"].to(device)
        if detach_phi_:
            with torch.no_grad():
                w = torch.softmax(phi_model(emb), dim=0)
        else:
            w = torch.softmax(phi_model(emb), dim=0)
        return torch.dot(w, per_sample)

    def _val_loss(q_head: QHead, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        per_sample = _forward_per_sample(q_head, batch)
        return per_sample.mean()

    # ---------------------------------------------------------------
    # WebShop simulation env + catalog goals
    # ---------------------------------------------------------------
    import web_agent_site.webshop_llm_utils as webshop_http
    if webshop_base_url is not None:
        webshop_http.WEBSHOP_URL = webshop_base_url.rstrip("/")
    webshop_env_mod = webshop_http.env
    from web_agent_site.experiment_goals import load_webshop_products_and_goals

    _, _, _, _, catalog_goals = load_webshop_products_and_goals()
    eval_name_set = set(eval_category_names)
    train_name_set = set(train_category_names)

    def _goal_cat(g):
        return "-".join(g.get("product_category", "unknown").split("\u203a")[:2])

    eval_goal_indices = [i for i, g in enumerate(catalog_goals) if _goal_cat(g) in eval_name_set]
    train_goal_indices = [i for i, g in enumerate(catalog_goals) if _goal_cat(g) in train_name_set]
    print(f"Catalog goals: {len(catalog_goals)};  eval-cat indices: {len(eval_goal_indices)},  train-cat indices: {len(train_goal_indices)}")

    # ---------------------------------------------------------------
    # Save helpers
    # ---------------------------------------------------------------
    if save_dir is not None:
        save_dir = os.path.expanduser(save_dir)
    elif outputs_path is not None:
        save_dir = os.path.join(os.path.expanduser(outputs_path), exp_name or "bilevel_run")
    saved_checkpoints: deque = deque(loop_state.get("saved_checkpoints", []))

    def _save(name: str, add_to_queue: bool = True, **extra):
        nonlocal saved_checkpoints
        if save_dir is None:
            return
        d = os.path.join(save_dir, name)
        os.makedirs(d, exist_ok=True)
        if add_to_queue and max_checkpoints is not None and len(saved_checkpoints) >= max_checkpoints:
            old = saved_checkpoints.popleft()
            if os.path.isdir(old):
                import shutil
                shutil.rmtree(old, ignore_errors=True)
        ckpt = {
            "model": {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k.lower()},
            "q_head_w": q_head_w.state_dict(),
            "q_head_u": q_head_u.state_dict(),
            "phi_model": phi_model.state_dict(),
            "opt_base": opt_base.state_dict(),
            "opt_w": opt_w.state_dict(),
            "opt_u": opt_u.state_dict(),
            "opt_phi": opt_phi.state_dict(),
            "loop_state": extra,
        }
        torch.save(ckpt, os.path.join(d, "checkpoint.pt"))
        if add_to_queue:
            saved_checkpoints.append(d)
        print(f"Saved checkpoint → {d}")

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------
    def evaluate(global_step_: int, **kwargs):
        model.eval()
        q_head_w.eval()
        phi_model.eval()

        # ---- Losses on eval / train / val ----
        loss_accum = {"eval": 0.0, "train": 0.0, "val": 0.0}
        for tag, ds in [("eval", eval_ds), ("train", train_ds), ("val", val_ds)]:
            loader = DataLoader(ds, batch_size=eval_loss_bsize, shuffle=False,
                                collate_fn=mc_collate_fn, drop_last=True)
            c = 0
            for batch in loader:
                if c >= eval_loss_batches:
                    break
                with torch.no_grad():
                    l = _val_loss(q_head_w, batch)
                loss_accum[tag] += l.item()
                c += 1
            if c > 0:
                loss_accum[tag] /= c

        phi_avg, phi_summary = compute_phi_avg_weight_by_category(
            phi_model, train_embeddings, train_categories, partition_display_names, device,
        )
        print(f"Phi avg weight by category: {phi_avg}")
        print(f"Phi weight summary: {phi_summary}")
        print(f"Eval loss: {loss_accum['eval']:.4f}  Train loss: {loss_accum['train']:.4f}  Val loss: {loss_accum['val']:.4f}")

        # ---- WebShop simulation rollouts ----
        interaction_summary = None
        train_interaction_summary = None
        eval_interaction_logs = None
        train_interaction_logs = None

        if not skip_eval_simulation:
            try:
                requests.get(webshop_http.WEBSHOP_URL, timeout=5)
            except Exception as e:
                print(f"WebShop sim skipped (server at {webshop_http.WEBSHOP_URL}?): {e}")
                interaction_summary = {"webshop_unreachable": True, "error": str(e)}
                train_interaction_summary = interaction_summary
            else:
                newline_tok = tokenizer.encode("\n", add_special_tokens=False)
                eos_id = newline_tok[0] if newline_tok else tokenizer.eos_token_id
                gen_cfg = GenerationConfig(
                    do_sample=policy_do_sample,
                    temperature=policy_temperature,
                    top_p=policy_top_p,
                    top_k=policy_top_k,
                    eos_token_id=eos_id,
                    pad_token_id=pad_token_id,
                    max_new_tokens=policy_max_output_length,
                )
                policy = SimplePolicy(model, tokenizer, gen_cfg, policy_max_input_length, device,
                                      system_context=WEBSHOP_SYSTEM)

                eval_mean, _, eval_interaction_logs = webshop_rollout_mean_reward(
                    policy, webshop_env_mod, catalog_goals, eval_goal_indices,
                    policy_n_rollouts, webshop_sim_max_steps, "mc_eval",
                    webshop_prompt_truncate_chars,
                )
                train_mean, _, train_interaction_logs = webshop_rollout_mean_reward(
                    policy, webshop_env_mod, catalog_goals, train_goal_indices,
                    policy_n_rollouts, webshop_sim_max_steps, "mc_train",
                    webshop_prompt_truncate_chars,
                )
                interaction_summary = {"mean_reward_eval_categories": eval_mean, "n_rollouts": policy_n_rollouts}
                train_interaction_summary = {"mean_reward_train_categories": train_mean, "n_rollouts": policy_n_rollouts}
                print(f"WebShop sim: eval-cat reward={eval_mean:.4f},  train-cat reward={train_mean:.4f}")

                # full conversation dump
                for label, logs in [("EVAL", eval_interaction_logs), ("TRAIN", train_interaction_logs)]:
                    print("=" * 80)
                    print(f"FULL CONVERSATIONS - {label}-CATEGORY GOALS")
                    print("=" * 80)
                    for idx_log, inter in enumerate(logs or []):
                        print(f"\n{label} Conversation {idx_log + 1}:")
                        print("-" * 80)
                        print(f"Goal idx: {inter['goal_idx']} | Final reward: {inter['final_reward']:.4f}")
                        print(f"Instruction: {inter['instruction']}")
                        for item in inter.get("log", []):
                            print(f"  Step {item['step']}  Action: {item['action']}")
                            print(f"  Observation: {item['observation']}")
                            print(f"  Reward: {item['reward']:.4f}  Done: {item['done']}")
                    print("=" * 80)

        eval_log = {
            "eval_loss": loss_accum["eval"],
            "train_loss": loss_accum["train"],
            "val_loss": loss_accum["val"],
            "generation_metrics": interaction_summary,
            "train_generation_metrics": train_interaction_summary,
            "phi_avg_weight_by_category": phi_avg,
            **phi_summary,
            **kwargs,
        }
        if use_wandb and is_main_process:
            import wandb
            wandb.log({
                "eval_loss": loss_accum["eval"],
                "train_loss": loss_accum["train"],
                "val_loss": loss_accum["val"],
                **phi_summary,
                **{f"phi_avg_weight/{k}": v for k, v in phi_avg.items()},
                "global_step": global_step_,
            })

        model.train()
        q_head_w.train()
        phi_model.train()
        return loss_accum["val"], eval_log

    # ---------------------------------------------------------------
    # Main bilevel training loop
    # ---------------------------------------------------------------
    alpha = init_alpha
    global_step = int(loop_state.get("step", 0))
    best_val_loss = float(loop_state.get("best_val_loss", float("inf")))
    start_outer = int(loop_state.get("outer_iter", -1)) + 1
    if resume_from_checkpoint is not None and start_outer > 0:
        alpha = init_alpha * (tau ** start_outer)
        print(f"Resuming: outer_iter={start_outer - 1}, step={global_step}, alpha={alpha:.4f}")

    loss_L1 = 0.0
    loss_w_L2 = 0.0
    loss_u_L2 = 0.0

    model.train()
    q_head_u.train()
    q_head_w.train()
    phi_model.train()

    for i in tqdm.tqdm(range(start_outer, num_outer_iter), desc="outer"):
        alpha *= tau
        t_outer = time.time()
        print(f"\nOuter iteration {i + 1}/{num_outer_iter},  alpha={alpha:.4f}")

        for k in range(num_inner_iter):
            if k % 50 == 0:
                print(f"  inner {k}/{num_inner_iter}")

            # ============================================================
            # Phase (a): update u  (LoRA + q_head_u, weighted train loss)
            # ============================================================
            t_a = time.time()
            if init_alpha > 0:
                opt_base.zero_grad()
                opt_u.zero_grad()
                for _ga in range(grad_accum_steps):
                    for _ in range(inner_opt_steps):
                        batch = next(train_iter)
                        loss_a = _weighted_train_loss(q_head_u, batch, detach_phi_=True) / grad_accum_steps
                        loss_a.backward()
                        loss_u_L2 = loss_a.item() * grad_accum_steps
                opt_base.step()
                opt_u.step()
            if do_time:
                print(f"    Phase (a) {time.time() - t_a:.2f}s")

            # ============================================================
            # Phase (b): sync  q_u → q_w
            # ============================================================
            t_b = time.time()
            if init_alpha > 0:
                q_head_w.load_state_dict(q_head_u.state_dict())
                opt_w = torch.optim.AdamW(
                    q_head_w.parameters(), lr=lr_q_w,
                    betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay,
                )
            if do_time:
                print(f"    Phase (b) {time.time() - t_b:.2f}s")

            # ============================================================
            # Phase (c): update w  (train + val)
            # ============================================================
            t_c = time.time()
            opt_base.zero_grad()
            opt_w.zero_grad()
            for _ga in range(grad_accum_steps):
                for _ in range(inner_opt_steps):
                    train_batch = next(train_iter)
                    if init_alpha > 0:
                        val_batch = next(val_iter)
                        # (c-i) weighted train loss
                        loss_c_train = _weighted_train_loss(q_head_w, train_batch, detach_phi_=True)
                        (loss_c_train / grad_accum_steps).backward()
                        loss_w_L2 = loss_c_train.item()

                        # (c-ii) val loss  (scaled by alpha)
                        opt_base.step(); opt_w.step()
                        opt_base.zero_grad(); opt_w.zero_grad()
                        loss_c_val = _val_loss(q_head_w, val_batch) * alpha
                        (loss_c_val / grad_accum_steps).backward()
                        loss_L1 = loss_c_val.item()
                    else:
                        loss_c = _val_loss(q_head_w, train_batch)
                        (loss_c / grad_accum_steps).backward()
                        loss_L1 = loss_c.item()
                        if global_step % log_every == 0:
                            print(f"  step {global_step}  loss={loss_L1:.4f}")
            opt_base.step()
            opt_w.step()
            if do_time:
                print(f"    Phase (c) {time.time() - t_c:.2f}s")

            # ============================================================
            # Phase (d): update φ  (weight predictor)
            # ============================================================
            t_d = time.time()
            if init_alpha > 0 and val_split > 0:
                phi_steps = max(1, int(inner_opt_steps / phi_update_factor))
                opt_phi.zero_grad()
                for _ in range(phi_steps):
                    batch_d = next(train_iter)
                    input_ids = batch_d["input_ids"].to(device)
                    attn_mask = (input_ids != pad_token_id).long()
                    sta = batch_d["should_take_action"].to(device)
                    rets = batch_d["returns"].to(device)
                    emb_d = batch_d["embedding"].to(device)

                    with torch.no_grad():
                        with autocast_ctx():
                            out_d = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                            hidden_d = out_d.hidden_states[-1]

                            qu_out = q_head_u(hidden_d).float()
                            qw_out = q_head_w(hidden_d).float()

                        qu = torch.gather(qu_out[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                        qw = torch.gather(qw_out[:, :-1], 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

                        psl_u = mc_per_sample_loss(qu, qu_out[:, :-1], input_ids[:, 1:],
                                                   attn_mask[:, 1:], sta[:, 1:], rets[:, 1:], cql_weight)
                        psl_w = mc_per_sample_loss(qw, qw_out[:, :-1], input_ids[:, 1:],
                                                   attn_mask[:, 1:], sta[:, 1:], rets[:, 1:], cql_weight)

                    logits_phi = phi_model(emb_d)
                    weights_phi = torch.softmax(logits_phi, dim=0)
                    loss_phi = torch.dot(weights_phi, psl_w) - torch.dot(weights_phi, psl_u)
                    loss_phi.backward()
                opt_phi.step()
            if do_time:
                print(f"    Phase (d) {time.time() - t_d:.2f}s")

            global_step += inner_opt_steps

        # ---- End of outer iteration: evaluate & save ----
        val_loss, eval_log = evaluate(
            global_step,
            loss_L1=loss_L1, loss_w_L2=loss_w_L2, loss_u_L2=loss_u_L2,
        )
        best_val_loss = min(best_val_loss, val_loss)

        _save(
            name=f"outer_iter_{global_step}",
            add_to_queue=True,
            best_val_loss=best_val_loss,
            step=global_step,
            outer_iter=i,
            saved_checkpoints=list(saved_checkpoints),
            eval_log=eval_log,
        )

        if do_time:
            print(f"Outer iteration {i + 1} took {time.time() - t_outer:.2f}s")

    print("Training complete.")


if __name__ == "__main__":
    tyro.cli(main)
