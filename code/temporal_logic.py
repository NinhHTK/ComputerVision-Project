"""Temporal logic shared by realtime detection and video evaluation."""

from dataclasses import dataclass


@dataclass
class DurationTracker:
    """Track a condition that must remain true for a continuous duration."""

    min_duration_seconds: float
    elapsed_seconds: float = 0.0
    active: bool = False
    just_activated: bool = False

    def __post_init__(self):
        if self.min_duration_seconds <= 0:
            raise ValueError("min_duration_seconds must be greater than 0")

    def reset(self):
        self.elapsed_seconds = 0.0
        self.active = False
        self.just_activated = False

    def update(self, condition, delta_seconds):
        """
        Update state using the duration of the current sample/frame.

        `just_activated` is true only once when the duration threshold is first
        reached. When condition=False, the continuous sequence is reset.
        """
        self.just_activated = False

        if not condition:
            self.reset()
            return False

        self.elapsed_seconds += max(0.0, float(delta_seconds))
        reached = self.elapsed_seconds + 1e-9 >= self.min_duration_seconds

        if reached and not self.active:
            self.active = True
            self.just_activated = True

        return self.active
