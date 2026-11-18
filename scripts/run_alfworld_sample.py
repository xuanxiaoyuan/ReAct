"""Small ALFWorld ReAct run using the original examples and a chat API.

Requires: pip install alfworld pyyaml; ALFWORLD_DATA and DASHSCOPE_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from react_agent.client import OpenAICompatibleClient


PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=Path(r"D:\learn\ReAct-original"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=49)
    parser.add_argument("--output", type=Path, default=Path("trajs/alfworld_sample.jsonl"))
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")
    if not os.getenv("ALFWORLD_DATA"):
        parser.error("set ALFWORLD_DATA to the downloaded game directory")

    import yaml
    import alfworld.agents.environment

    config = yaml.safe_load((args.original / "base_config.yaml").read_text(encoding="utf-8"))
    config["general"]["use_cuda"] = False
    examples = json.loads((args.original / "prompts" / "alfworld_3prompts.json").read_text(encoding="utf-8"))
    env = alfworld.agents.environment.AlfredTWEnv(config, train_eval="eval_out_of_distribution")
    env = env.init_env(batch_size=1)
    model = OpenAICompatibleClient.from_env()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("a", encoding="utf-8") as output:
        for episode in range(args.episodes):
            observation, info = env.reset()
            task = "\n".join(observation[0].split("\n\n")[1:])
            gamefile = info["extra.gamefile"][0]
            category = next((value for key, value in PREFIXES.items() if
                             Path(gamefile).parts[-3].startswith(key)), None)
            if category is None:
                raise RuntimeError(f"unknown ALFWorld task type: {gamefile}")
            prompt = ("Interact with a household to solve a task. Here are two examples.\n"
                      + examples[f"react_{category}_1"] + examples[f"react_{category}_0"]
                      + "\nHere is the task.\n" + task + "\n>")
            trajectory = []
            won = False
            for step in range(1, args.max_steps + 1):
                reply = model.complete([{"role": "user", "content": prompt}], [])
                action = (reply.get("content") or "").strip().splitlines()[0].strip()
                if not action:
                    raise RuntimeError(f"empty model action in episode {episode}, step {step}")
                next_observation, _, done, step_info = env.step([action])
                observed = "OK." if action.startswith("think:") else next_observation[0]
                won = bool(step_info["won"][0])
                trajectory.append({"step": step, "action": action, "observation": observed,
                                   "done": bool(done[0]), "won": won})
                prompt += f" {action}\n{observed}\n>"
                if done[0]:
                    break
            record = {"episode": episode, "gamefile": gamefile, "task_type": category,
                      "task": task, "model": model.model, "won": won, "trajectory": trajectory}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"episode={episode} won={won} steps={len(trajectory)}")


if __name__ == "__main__":
    main()
