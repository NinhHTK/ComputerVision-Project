"""Logic thời gian dùng chung cho realtime và video evaluation."""

from dataclasses import dataclass


@dataclass
class DurationTracker:
    """Theo dõi một điều kiện phải đúng liên tục trong một khoảng thời gian."""

    min_duration_seconds: float
    elapsed_seconds: float = 0.0
    active: bool = False
    just_activated: bool = False

    def __post_init__(self):
        if self.min_duration_seconds <= 0:
            raise ValueError("min_duration_seconds phải lớn hơn 0")

    def reset(self):
        self.elapsed_seconds = 0.0
        self.active = False
        self.just_activated = False

    def update(self, condition, delta_seconds):
        """
        Cập nhật trạng thái với thời lượng của mẫu/frame hiện tại.

        `just_activated` chỉ True đúng một lần khi điều kiện lần đầu đạt
        ngưỡng thời gian. Khi condition=False, chuỗi liên tục bị reset.
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
