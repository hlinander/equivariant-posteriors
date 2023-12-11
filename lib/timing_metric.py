import time
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class Timing:
    data: Dict[str, List[float]] = field(default_factory=lambda: dict())
    starts: Dict[str, float] = field(default_factory=lambda: dict())

    def start(self, name):
        self.starts[name] = time.time()

    def stop(self, name):
        stop = time.time()
        if name not in self.data:
            self.data[name] = []
        if name in self.starts:
            self.data[name].append(stop - self.starts[name])

    def serialize(self):
        return self.data

    def deserialize(self, serialized):
        self.data = serialized
