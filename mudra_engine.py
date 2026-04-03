from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


FingerState = Tuple[bool, bool, bool, bool, bool]


@dataclass
class MudraEvent:
    mudra: str
    frame_index: int


@dataclass
class JutsuMatch:
    name: str
    sequence: Sequence[str]


class MudraClassifier:
    """Classifies a hand pose into a simplified mudra label based on finger states."""

    MUDRA_RULES: Dict[FingerState, str] = {
        (False, False, False, False, False): "TIGER",
        (False, True, False, False, False): "RAM",
        (False, True, True, False, False): "BIRD",
        (False, True, True, True, False): "HORSE",
        (True, True, False, False, False): "DOG",
        (True, True, True, True, True): "BOAR",
    }

    def classify(self, finger_state: FingerState) -> Optional[str]:
        return self.MUDRA_RULES.get(finger_state)


class JutsuComposer:
    """Tracks mudra events and detects known jutsu sequences."""

    def __init__(self) -> None:
        self.jutsu_library: Dict[str, List[str]] = {
            "Fireball (Demo)": ["DOG", "TIGER", "BOAR", "RAM"],
            "Water Dragon (Demo)": ["BOAR", "BIRD", "RAM", "HORSE"],
            "Shadow Clone (Demo)": ["RAM", "BIRD", "DOG"],
        }
        self.events: List[MudraEvent] = []

    def add_mudra(self, mudra: str, frame_index: int) -> Optional[JutsuMatch]:
        self.events.append(MudraEvent(mudra=mudra, frame_index=frame_index))

        # Keep history small and fast while still allowing sequence detection.
        if len(self.events) > 30:
            self.events = self.events[-30:]

        symbols = [event.mudra for event in self.events]

        for name, sequence in self.jutsu_library.items():
            if len(symbols) >= len(sequence) and symbols[-len(sequence) :] == sequence:
                return JutsuMatch(name=name, sequence=sequence)

        return None

    def recent_sequence(self, max_items: int = 8) -> List[str]:
        return [event.mudra for event in self.events[-max_items:]]
