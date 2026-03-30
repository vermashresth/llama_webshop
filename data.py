"""
Offline WebShop trajectories for MC training (same transcript shape as llm_interact.run_episode).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from LLM_RL.environment import Text, TextTrajectory

# Mirrors web_agent_site/envs/llm_interact.py (in-context exemplar + rules).
EXAMPLE_TRAJECTORY = """
Instruction: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars

[Search]
Action: search[bright citrus deodorant]
Observation: [B078GWRC1J] Calming Citrus Deodorant $80.00 [B09M63B87V] Strong Citrus Scent $45.00 [B078GWRC1J] ...
Action: click[B09M63B87V]
Observation: [< Prev] smell [lavender] [citrus] size [3 ounce (pack of 1)] [12 ounce (pack of 3)]... [Buy Now]
Action: click[citrus]
Observation: You have clicked citrus.
Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).
Action: click[Buy Now]
"""

WEBSHOP_PROMPT = """Webshop

You act as a shopping agent. Given an instruction and observations, output ONE action.

Actions:
search[query]  # 4-8 words, core keywords only
click[item]    # click product ID or option

Rules:
- Only ONE search allowed.
- After search, pick best item (price + attributes).
- On item page: select attributes matching instruction, then click[Buy Now].
- If item is wrong, click[< Prev] and choose another.

Example:
Instruction: 3 ounce bright citrus deodorant sensitive skin under 50 dollars

[Search]
Action: search[bright citrus deodorant sensitive]
Observation: [A] Citrus Deodorant $80 [B] Citrus Sensitive $45
Action: click[B]
Observation: size [3 ounce] scent [citrus] [Buy Now]
Action: click[3 ounce]
Observation: selected
Action: click[citrus]
Observation: selected
Action: click[Buy Now]

Now begin.

Instruction: {instruction}
"""


def build_init_prompt(instruction_text: str) -> str:
    # return f"Webshop\nInstruction: {instruction_text}\n\n{EXAMPLE_TRAJECTORY.strip()}"
    return WEBSHOP_PROMPT.format(instruction=instruction_text)


def load_webshop_offline_json(path: str) -> List[Dict[str, Any]]:
    print(f"Loading webshop offline JSON from {path}")
    """Load JSON as list of rollout dicts. Supports {\"all_results\": [...]} or a bare list."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "all_results" in obj:
        return list(obj["all_results"])
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Expected list or dict with 'all_results', got {type(obj)}")


def get_category_name_from_result(result: Dict[str, Any]) -> str:
    goal = result.get("goal") or {}
    # cat = goal.get("category", "unknown")
    cat = '-'.join(goal.get("product_category", "unknown").split('›')[:2])
    return str(cat).strip() if cat is not None else "unknown"


def filter_results_by_categories_with_idx(
    results: List[Dict[str, Any]],
    allowed_category_names: List[str],
) -> Tuple[List[Dict[str, Any]], List[int]]:
    allowed = set(allowed_category_names)
    out, idx = [], []
    for i, r in enumerate(results):
        if get_category_name_from_result(r) in allowed:
            out.append(r)
            idx.append(i)
    return out, idx


def create_trajectory_from_offline_result(result: Dict[str, Any]) -> TextTrajectory:
    """
    Build TextTrajectory from one offline rollout (keys: goal, log, reward, ...).
    Transcript matches llm_interact.run_episode suffix construction.
    """
    goal = result.get("goal") or {}
    instruction = result.get("instruction") or goal.get("instruction_text", "")
    init_prompt = build_init_prompt(instruction)
    log = result.get("log") or []
    terminal_reward = float(result.get("reward", 0.0))

    text_history: List[Text] = []
    reward: List[float] = []

    reset_i = next((k for k, e in enumerate(log) if e.get("action") == "reset"), None)
    if reset_i is None:
        raise ValueError("WebShop log must contain an action=='reset' entry")

    obs0 = log[reset_i]["observation"]
    prompt_suffix = f"{obs0}\n\nAction:"
    text_history.append(Text(init_prompt + prompt_suffix, is_action=False))
    reward.append(0.0)

    traj_done = any(bool(e.get("done", False)) for e in log)

    j = reset_i + 1
    while j < len(log):
        action = log[j].get("action", "")
        obs = log[j].get("observation", "")
        if action == "reset":
            j += 1
            continue

        is_terminal = bool(log[j].get("done", False))
        text_history.append(Text(action + "\n", is_action=True))
        reward.append(terminal_reward if is_terminal else 0.0)

        prompt_suffix = prompt_suffix + f" {action}\nObservation: {obs}\n\nAction:"
        text_history.append(Text(init_prompt + prompt_suffix, is_action=False))
        reward.append(0.0)
        j += 1

    return TextTrajectory(tuple(text_history), tuple(reward), traj_done)


def create_trajectories_from_offline_results(results: List[Dict[str, Any]]) -> List[TextTrajectory]:
    return [create_trajectory_from_offline_result(r) for r in results]
