"""A deterministic ALFWorld-style mini environment and its function tools."""

from __future__ import annotations

from dataclasses import dataclass, field

from .tools import ToolError, ToolRegistry, function_tool


@dataclass
class MiniHouse:
    room: str = "hall"
    inventory: list[str] = field(default_factory=list)
    locations: dict[str, str] = field(default_factory=lambda: {"apple": "kitchen", "mug": "study"})
    dirty: set[str] = field(default_factory=lambda: {"mug"})
    containers: dict[str, list[str]] = field(default_factory=lambda: {"fridge": [], "cabinet": []})
    connections: dict[str, list[str]] = field(default_factory=lambda: {
        "hall": ["kitchen", "study"], "kitchen": ["hall"], "study": ["hall"]
    })
    finished: bool = False

    def registry(self) -> ToolRegistry:
        house = self

        @function_tool
        def observe() -> str:
            """Inspect the current room, visible objects, exits, and inventory."""
            visible = sorted(item for item, place in house.locations.items() if place == house.room)
            return (
                f"room={house.room}; exits={house.connections[house.room]}; "
                f"visible={visible}; inventory={house.inventory}; dirty_visible={sorted(house.dirty & set(visible))}"
            )

        @function_tool
        def go(room: str) -> str:
            """Move through an exit to an adjacent room.
            room: Name of an adjacent room.
            """
            if room not in house.connections[house.room]:
                raise ToolError(f"cannot go from {house.room} to {room}; exits={house.connections[house.room]}")
            house.room = room
            return f"moved to {room}"

        @function_tool
        def take(item: str) -> str:
            """Pick up a visible object.
            item: Object to take.
            """
            if house.locations.get(item) != house.room:
                raise ToolError(f"{item} is not visible in {house.room}")
            house.locations.pop(item)
            house.inventory.append(item)
            return f"picked up {item}"

        @function_tool
        def clean(item: str) -> str:
            """Clean a held object; a sink is available in the kitchen.
            item: Held object to clean.
            """
            if house.room != "kitchen":
                raise ToolError("cleaning is only possible in the kitchen")
            if item not in house.inventory:
                raise ToolError(f"not holding {item}")
            house.dirty.discard(item)
            return f"cleaned {item}"

        @function_tool
        def put(item: str, container: str) -> str:
            """Put a held object into a container in the current room.
            item: Held object to put away.
            container: Target container, fridge or cabinet.
            """
            expected_room = {"fridge": "kitchen", "cabinet": "study"}
            if container not in expected_room:
                raise ToolError(f"unknown container: {container}")
            if house.room != expected_room[container]:
                raise ToolError(f"{container} is in {expected_room[container]}, not {house.room}")
            if item not in house.inventory:
                raise ToolError(f"not holding {item}")
            house.inventory.remove(item)
            house.containers[container].append(item)
            return f"put {item} in {container}"

        @function_tool
        def finish(answer: str) -> str:
            """End the run after completing the task.
            answer: Concise final report.
            """
            if not house.task_succeeded():
                raise ToolError(
                    "task is not complete; required: apple in fridge, clean mug in study cabinet"
                )
            house.finished = True
            return answer

        return ToolRegistry([observe, go, take, clean, put, finish])

    def task_succeeded(self) -> bool:
        return "apple" in self.containers["fridge"] and "mug" in self.containers["cabinet"] and "mug" not in self.dirty


DEFAULT_TASK = "Put the apple in the fridge. Then clean the mug and put it in the study cabinet."
