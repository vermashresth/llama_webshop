#!/usr/bin/env python3
"""
WebShop LLM experiment: interact with local WebShop site via LLM, compute average
reward over repeated runs per goal, across multiple goals, and report cost.

Prerequisites:
  - WebShop server running on localhost:3000 (e.g. ./run_dev.sh)
  - Run from repo root or from web/WebShop with PYTHONPATH including the app

Usage:
  cd web/WebShop && python run_llm_experiment.py [--max-steps 15] [--num-goals 20] [--runs-per-goal 20]
"""

import argparse
import json
import random
import re
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, '../..')

import requests

from web_agent_site.webshop_llm_utils import env as webshop_env, WEBSHOP_URL



def load_goals_for_experiment():
    """Synthetic goals + shuffle seed 233; same ordering as Flask (see experiment_goals.py)."""
    from web_agent_site.experiment_goals import load_webshop_products_and_goals

    _, _, _, _, goals = load_webshop_products_and_goals()
    print(f"Loaded {len(goals)} shuffled experiment goals (same list as WebShop server).")
    return goals





# -------- LLM client (HUIT API with cost) --------
def invoke_huit(
    messages,
    model="gpt-4o-mini",
    max_tokens=256,
    temperature=0.5,
    api_key=None,
    endpoint_url="https://go.apis.huit.harvard.edu/ais-openai-direct/v1/chat/completions",
    max_attempts=3,
    wait_seconds=60,
):
    """
    Send chat request to HUIT OpenAI endpoint. Returns (content, cost).
    cost is taken from result_json['your_cost_this_transaction'] if present, else 0.
    messages: list of {"role": "user"|"system"|"assistant", "content": str}
    """
    payload = {
        "model": model.replace("-huit", ""),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
    }
    headers = {"Content-Type": "application/json", "api-key": api_key or "C5GQr9A03fTwUH8eiTlRGu7IPmRRSo9vpQGPp5yeVOlchQcv"}

    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(endpoint_url, headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            result = r.json()
            content = result["choices"][0]["message"]["content"]
            cost = float(result.get("your_cost_this_transaction", 0.0))
            return content, cost
        except requests.exceptions.RequestException as e:
            warnings.warn(f"HUIT attempt {attempt}/{max_attempts} failed: {e}")
            if attempt == max_attempts:
                raise
            time.sleep(wait_seconds)
    return "", 0.0


# -------- Prompt and action parsing --------
# WEBSHOP_SYSTEM = """You are playing Webshop. You will see a text view of the current page and must output exactly one action per turn.

# Actions:
# - search[query] — type a search query (only from the start page).
# - click[button or option] — click a button or option shown in brackets, e.g. click[Buy Now], click[B078GWRC1J], click[bright citrus].
# - think[your reasoning] — optional; no effect on the page.

# Output only one action per message, in the form action[argument]. Nothing else."""

# WEBSHOP_SYSTEM = """You are playing Webshop. You will see a text view of the current page and must output exactly one action per turn.

# CRITICAL RULES:
# - search[query] — type a search query. You can ONLY use this if you see [Search] on the page. 
# - click[button or option] — click a button or option shown in brackets. 
# - If you want to search again but are on a results page, you MUST first output click[Back to Search]. Do not use search[] until you are back on the main page.
# - think[your reasoning] — optional; no effect on the page.

# Output only one action per message, in the form action[argument]. Nothing else."""

WEBSHOP_SYSTEM = """You are playing Webshop. You will see a text view of the current page and must output exactly one action per turn.

CRITICAL RULES:
- search[query] — type a search query. Your query MUST be short (4 to 8 words maximum). Extract only the core keywords (e.g., "mens tuxedo shirt"). 
- You only get ONE search. Once you see the search results, you MUST choose the product with the best price and attributes matching the instruction and use click[].
- Once you see details of an item, you can click on attributes relevant to instruction and click[Buy Now]
- If the attributes or price don't match the instruction and you think there is a better item, you can click[< Prev] to go back to the previous page and click on the next item.

Actions:
- search[query] — search for the core item.
- click[button or option] — click a button or option shown in brackets, e.g. click[Buy Now], click[B078GWRC1J].

Output only one action per message, in the form action[argument]. Nothing else."""

# EXAMPLE_TRAJECTORY = """
# Instruction: i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars

# [Search]
# Action: search[3 ounce bright citrus deodorant sensitive skin]
# Observation: [B078GWRC1J] Calming Citrus Deodorant $80.00 [B09M63B87V] Strong Citrus Scent $45.00 [B078GWRC1J] ...
# Action: click[B09M63B87V]
# Observation: [< Prev] smell [lavender] [citrus] size [3 ounce (pack of 1)] [12 ounce (pack of 3)]... [Buy Now]
# Action: click[citrus]
# Observation: You have clicked citrus.
# Action: click[3 ounce (pack of 1)]
# Observation: You have clicked 3 ounce (pack of 1).
# Action: click[Buy Now]
# """

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

Instruction: Find me machine wash women's tops, tees & blouses with short sleeve, unique design, relaxed fit, button closure, polyester spandex with color: blue, and size: xx-large, and price lower than 50.00 dollars 

[Search]
Action: search[womens blue tops short sleeve relaxed]
Observation: [B08G8DXR5N] Beninos Sports Bras for Women... $6.99 [B09M63B87V] YALFJV Women Long Sleeve Crew Neck Side Button T Shirts Tunic... $10.71 to $18.34 [B09QCVCYVY] Burband Womens High Waist Bike Shorts... $6.82
Action: click[B08G8DXR5N]
Observation: [< Prev] Beninos Sports Bras for Women - Activewear Tops for Yoga Running Fitness... Price: $6.99 Rating: N.A. [Buy Now]
Action: think[This is a sports bra, which doesn't match the request for short sleeves and button closure. I should go back and check the next item.]
Action: click[< Prev]
Observation: [B08G8DXR5N] Beninos Sports Bras for Women... $6.99 [B09M63B87V] YALFJV Women Long Sleeve Crew Neck Side Button T Shirts Tunic... $10.71 to $18.34 [B09QCVCYVY] Burband Womens High Waist Bike Shorts... $6.82
Action: click[B09M63B87V]
Observation: [< Prev] color [xnj-tshirt324-beige][xnj-tshirt325-black][xnj-tshirt326-blue][xnj-tshirt327-green]... size [small][medium][large][x-large][xx-large] Price: $10.71 to $18.34 Rating: N.A. [Buy Now]
Action: click[xnj-tshirt326-blue]
Observation: You have clicked xnj-tshirt326-blue.
Action: click[xx-large]
Observation: You have clicked xx-large.
Action: click[Buy Now]
"""


def build_init_prompt(instruction_text: str) -> str:
    return f"Webshop\nInstruction: {instruction_text}\n\n" + EXAMPLE_TRAJECTORY.strip()


def parse_action_from_llm(text: str):
    """Extract a single action from model output: search[...], click[...], or think[...]."""
    text = (text or "").strip()
    # Prefer first line that looks like action[arg]
    for line in text.split("\n"):
        line = line.strip()
        if re.match(r"^(search|click|think)\s*\[", line, re.IGNORECASE):
            # Normalize to exactly action[arg]
            m = re.match(r"(search|click|think)\s*\[(.*)\]", line, re.IGNORECASE | re.DOTALL)
            if m:
                kind, arg = m.group(1).lower(), m.group(2).strip()
                return f"{kind}[{arg}]"
            return line
    return None



import tqdm
# -------- Episode --------
def run_episode(
    session_id: str,
    instruction_text: str,
    llm_invoke,
    max_steps: int = 15,
    prompt_truncate_chars: int = 6000,
    verbose: bool = False,
):
    """
    Run one WebShop episode. Returns (reward, steps_used, total_cost, invalid_count).
    """
    init_prompt = build_init_prompt(instruction_text)
    prompt_suffix = ""
    total_cost = 0.0
    steps_used = 0
    invalid_count = 0
    action = "reset"
    log = []

    for step in tqdm.tqdm(range(max_steps + 1), desc="Steps"):
        try:
            observation, reward, done = webshop_env.step(session_id, action)
            prev_good_observation = observation
        except AssertionError:
            if 'search' in action.lower():
                observation = (
                    f"ERROR: Your action {action} is invalid here. "
                    f"HINT: You can only use 'search[]' once. "
                    f"you must choose an item from the search results.\n"
                )
            elif 'prev' in action.lower():
                observation = (
                    f"ERROR: Your action {action} is invalid here. "
                    f"HINT: You must click[] on an item from the search results to see product page. "
                )
            else:
                observation = (
                    f"ERROR: Your action {action} is invalid here. "
                    f"HINT: try something else! "
                )
            # observation = f"Your action {action} is invalid in the current webpage, try something else! Here is the current webpage: {prev_good_observation}"
            reward = 0.0
            done = False
            invalid_count += 1

        if action.startswith("think["):
            observation = "OK"

        if verbose:
            print(f"  Step {step} Action: {action}\n  Obs: {observation[:200]}...")
        log.append({
            "step": step,
            "action": action,
            "observation": observation,
            "reward": reward,
            "done": done,
        })

        if step > 0:
            prompt_suffix += f" {action}\nObservation: {observation}\n\nAction:"
        else:
            prompt_suffix += f"{observation}\n\nAction:"

        if done:
            return reward, steps_used, total_cost, invalid_count, log

        if step >= max_steps:
            return reward, steps_used, total_cost, invalid_count, log

        # Truncate to stay within context
        full_prompt = init_prompt + prompt_suffix[-(prompt_truncate_chars - len(init_prompt)) :]
        messages = [
            {"role": "system", "content": WEBSHOP_SYSTEM},
            {"role": "user", "content": full_prompt},
        ]
        content, cost = llm_invoke(messages)
        total_cost += cost
        steps_used += 1

        parsed = parse_action_from_llm(content)
        if parsed is None:
            # Fallback: try to fix common formats
            content_clean = content.strip().split("\n")[0].strip()
            if content_clean.startswith("search[") or content_clean.startswith("click[") or content_clean.startswith("think["):
                parsed = content_clean
            else:
                parsed = "think[no valid action]"
                invalid_count += 1
        action = parsed

    return reward, steps_used, total_cost, invalid_count, log
import math

# -------- Main experiment --------
def main(
        goal_indices,
        max_steps = 15,
        runs_per_goal = 3,
        model = "gpt-4o-mini",
        verbose = True,
        output = None
    ):
    # Check server
    try:
        r = requests.get(WEBSHOP_URL, timeout=5)
    except Exception as e:
        print(f"Error: Cannot reach WebShop at {WEBSHOP_URL}. Start the server first (e.g. ./run_dev.sh).", file=sys.stderr)
        sys.exit(1)

    def llm_invoke(messages):
        return invoke_huit(messages, model=model, temperature=0.0, max_tokens=256)

    print("Loading goals (same as server)...")
    # Order is fixed at seed 233 in web_agent_site.experiment_goals (must match Flask).
    goals = load_goals_for_experiment()
    bad_idx = [i for i in goal_indices if not (0 <= i < len(goals))]
    if bad_idx:
        raise IndexError(
            f"goal_indices out of range: {bad_idx}. "
            f"Valid indices for this catalog: 0..{len(goals) - 1} ({len(goals)} goals after shuffle)."
        )
    print(f"Catalog has {len(goals)} goals; running {len(goal_indices)} indices.")

    all_results = []
    total_cost = 0.0
    for goal_idx in tqdm.tqdm(goal_indices, desc="Goals", total=len(goal_indices)):
    # for goal_idx, goal in tqdm.tqdm(enumerate(goals), desc="Goals", total=len(goals)):
        goal = goals[goal_idx]
        instruction = goal["instruction_text"]
        print(f'My loop Goal: {goal_idx}', goal['instruction_text'])
        rewards = []
        goal_costs = []
        for run_idx in range(runs_per_goal):
            session_id = f"fixed_{goal_idx}_{run_idx}"
            reward, steps, cost, invalid, log = run_episode(
                session_id=session_id,
                instruction_text=instruction,
                llm_invoke=llm_invoke,
                max_steps=max_steps,
                verbose=verbose,
            )
            rewards.append(reward)
            goal_costs.append(cost)
            total_cost += cost
            all_results.append({
                "goal": goal,
                "goal_idx": goal_idx,
                "run_idx": run_idx,
                "instruction": instruction,
                "reward": reward,
                "steps": steps,
                "cost": cost,
                "invalid_actions": invalid,
                "log": log,
            })
        avg_reward = sum(rewards) / len(rewards)
        std_reward = math.sqrt(sum((r - avg_reward) ** 2 for r in rewards) / len(rewards))
        avg_cost_goal = sum(goal_costs) / len(goal_costs)
        print(f"Goal {goal_idx}: avg_reward={avg_reward:.4f} std_reward={std_reward:.4f} avg_cost={avg_cost_goal:.4f} (instruction: {instruction[:60]}...)")

    # Summary
    all_rewards = [x["reward"] for x in all_results]
    overall_avg = sum(all_rewards) / len(all_rewards)
    success_rate = sum(1 for r in all_rewards if r == 1.0) / len(all_rewards)
    per_goal_avg = []
    for goal_idx in goal_indices:
        goal_rewards = [x["reward"] for x in all_results if x["goal_idx"] == goal_idx]
        if goal_rewards:
            per_goal_avg.append((goal_idx, sum(goal_rewards) / len(goal_rewards)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"Indices run: {len(goal_indices)}  |  Catalog size: {len(goals)}  |  "
        f"Runs per goal: {runs_per_goal}  |  Max steps: {max_steps}"
    )
    print(f"Overall average reward: {overall_avg:.4f}")
    print(f"Success rate (reward=1.0): {success_rate:.2%}")
    print(f"Total API cost (your_cost_this_transaction): {total_cost:.4f}")
    print("\nPer-goal average reward:")
    for gidx, avg in per_goal_avg:
        short_inst = (goals[gidx]["instruction_text"][:50] + "…") if len(goals[gidx]["instruction_text"]) > 50 else goals[gidx]["instruction_text"]
        print(f"  Goal {gidx}: {avg:.4f}  — {short_inst}")

    out = {
        "config": {
            "max_steps": max_steps,
            "goal_indices": goal_indices,
            "num_goals_in_catalog": len(goals),
            "num_indices_run": len(goal_indices),
            "goals_shuffle_seed": 233,
            "runs_per_goal": runs_per_goal,
        },
        "summary": {"overall_avg_reward": overall_avg, "success_rate": success_rate, "total_cost": total_cost},
        "per_goal_avg": [{"goal_idx": g, "avg_reward": a} for g, a in per_goal_avg],
        "all_results": all_results,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDetailed results written to {output}")

import time
s=time.time()
name = 'clean_1200_1210_seed233_t0.5_1run.json'
main(goal_indices=list(range(1200, 1210)), runs_per_goal=1, verbose=False, output=name)
print('Total time taken:',time.time()-s)
with open(name, "r") as f:
    data = json.load(f)
print(sum(data["all_results"][i]['reward'] for i in range(50))/50)
print(sum(data["all_results"][i]['steps'] for i in range(50))/50)