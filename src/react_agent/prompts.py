"""Mode-specific instructions and executable few-shot trajectories."""

from __future__ import annotations

import json
from typing import Any

BASE = """Operate the household using function tools. Observations are actual tool results.
The example below is a demonstration; solve the NEW task at the end.
Call finish only after the environment goal is met. Use errors as observations."""
ACT_ONLY = BASE + "\nMode: Act-only. Call tools directly. Do not write thoughts or plans."
REACT = BASE + """
Mode: ReAct. Interleave a visible Thought, one Action, and its Observation.
Thought may plan, use common sense, or revise a plan after an error.
Put the Thought in assistant content accompanying the function call. One action per turn."""
REACT_IM = BASE + """
Mode: ReAct-IM. The runtime requests a separate inner monologue before EVERY action.
Describe only goal decomposition and the immediate subgoal based on the latest observation.
Do not speculate about locations or form a broad strategy. Then call one tool."""
PROMPTS = {"act-only": ACT_ONLY, "react": REACT, "react-im": REACT_IM}


def few_shot(mode: str) -> list[dict[str, Any]]:
    """A complete action/observation example on the same interface."""
    messages: list[dict[str, Any]] = [{"role": "user", "content":
        "Example task: Put the apple in the fridge. Then clean the mug and put it in the study cabinet."}]
    demonstration = [
        ("observe", {}, "room=hall; exits=['kitchen', 'study']; visible=[]; inventory=[]"),
        ("go", {"room": "kitchen"}, "moved to kitchen"),
        ("take", {"item": "apple"}, "picked up apple"),
        ("put", {"item": "apple", "container": "fridge"}, "put apple in fridge"),
        ("go", {"room": "hall"}, "moved to hall"),
        ("go", {"room": "study"}, "moved to study"),
        ("take", {"item": "mug"}, "picked up mug"),
        ("go", {"room": "hall"}, "moved to hall"),
        ("go", {"room": "kitchen"}, "moved to kitchen"),
        ("clean", {"item": "mug"}, "cleaned mug"),
        ("go", {"room": "hall"}, "moved to hall"),
        ("go", {"room": "study"}, "moved to study"),
        ("put", {"item": "mug", "container": "cabinet"}, "put mug in cabinet"),
        ("finish", {"answer": "Apple stored; mug cleaned and stored."}, "Apple stored; mug cleaned and stored."),
    ]
    react_thoughts = [
        "I need to locate the apple; inspect the room and exits.",
        "Nothing is visible. Food and a fridge are plausibly in the kitchen, so go there.",
        "The apple is in this room. Pick it up.",
        "I hold the apple and am in the kitchen; put it in the fridge.",
        "The apple is stored. I need the mug from the study, so return through the hall.",
        "Enter the study to find the mug.",
        "The mug is here; pick it up before cleaning.",
        "The sink is in the kitchen, reachable through the hall.",
        "Go to the kitchen to clean the held mug.",
        "Clean the mug using the kitchen sink.",
        "The clean mug belongs in the study cabinet; return via the hall.",
        "Enter the study with the clean mug.",
        "Place the clean mug in the cabinet.",
        "Both placements succeeded; report completion.",
    ]
    im_thoughts = [
        "Goal: apple in fridge. Immediate subgoal: inspect current room.",
        "Goal: apple in fridge. Immediate subgoal: reach the kitchen.",
        "Goal: apple in fridge. Immediate subgoal: take the apple.",
        "Goal: apple in fridge. Immediate subgoal: place held apple in fridge.",
        "Goal: store apple and clean/store mug. Immediate subgoal: reach hall.",
        "Goal: store apple and clean/store mug. Immediate subgoal: reach study.",
        "Goal: store apple and clean/store mug. Immediate subgoal: take mug.",
        "Goal: store apple and clean/store mug. Immediate subgoal: reach hall.",
        "Goal: store apple and clean/store mug. Immediate subgoal: reach kitchen.",
        "Goal: store apple and clean/store mug. Immediate subgoal: clean mug.",
        "Goal: store apple and clean/store mug. Immediate subgoal: reach hall.",
        "Goal: store apple and clean/store mug. Immediate subgoal: reach study.",
        "Goal: store apple and clean/store mug. Immediate subgoal: place mug.",
        "Goal: store apple and clean/store mug. Immediate subgoal: finish.",
    ]
    for index, (name, arguments, result) in enumerate(demonstration):
        identifier = f"example-{index}"
        if mode == "react-im":
            messages.append({"role": "assistant", "content": im_thoughts[index]})
        messages.append({"role": "assistant", "content": react_thoughts[index] if mode == "react" else None,
                         "tool_calls": [{"id": identifier, "type": "function", "function": {
                             "name": name, "arguments": json.dumps(arguments)}}]})
        messages.append({"role": "tool", "tool_call_id": identifier,
                         "content": json.dumps({"ok": True, "result": result})})
    # A second trajectory solves the same goal in a different order: mug first.
    messages.append({"role": "user", "content":
        "Example task 2: Clean the mug and put it in the study cabinet; also put the apple in the fridge."})
    second = [
        ("go", {"room": "study"}, "moved to study", "reach the mug in the study"),
        ("take", {"item": "mug"}, "picked up mug", "take the mug"),
        ("go", {"room": "hall"}, "moved to hall", "reach the hall"),
        ("go", {"room": "kitchen"}, "moved to kitchen", "reach the kitchen sink"),
        ("clean", {"item": "mug"}, "cleaned mug", "clean the mug"),
        ("take", {"item": "apple"}, "picked up apple", "take the apple"),
        ("put", {"item": "apple", "container": "fridge"}, "put apple in fridge", "store the apple"),
        ("go", {"room": "hall"}, "moved to hall", "reach the hall"),
        ("go", {"room": "study"}, "moved to study", "reach the study cabinet"),
        ("put", {"item": "mug", "container": "cabinet"}, "put mug in cabinet", "store the clean mug"),
        ("finish", {"answer": "Clean mug in cabinet; apple in fridge."},
         "Clean mug in cabinet; apple in fridge.", "finish the completed task"),
    ]
    for index, (name, arguments, result, subgoal) in enumerate(second):
        identifier = f"example-two-{index}"
        if mode == "react-im":
            messages.append({"role": "assistant", "content":
                f"Goal: clean/store mug and store apple. Immediate subgoal: {subgoal}."})
        thought = None
        if mode == "react":
            thought = f"The task is still in progress. Next I should {subgoal}."
        messages.append({"role": "assistant", "content": thought,
                         "tool_calls": [{"id": identifier, "type": "function", "function": {
                             "name": name, "arguments": json.dumps(arguments)}}]})
        messages.append({"role": "tool", "tool_call_id": identifier,
                         "content": json.dumps({"ok": True, "result": result})})
    return messages


def initial_messages(mode: str, task: str) -> list[dict[str, Any]]:
    return [{"role": "system", "content": PROMPTS[mode]}, *few_shot(mode),
            {"role": "user", "content": "NEW task: " + task}]
