"""B4 deterministic geometric queue-membership baseline."""

from __future__ import annotations

from dataclasses import dataclass

from research.features.schemas import PersonFeatures


@dataclass(frozen=True)
class RuleResult:
    is_queue: bool
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class GeometricRules:
    max_distance_to_path: float
    max_distance_to_counter: float
    min_distance_to_counter: float = 0.0
    require_neighbor: bool = False

    def predict(self, feature: PersonFeatures) -> RuleResult:
        checks = {
            "near_queue_path": feature.distance_to_path <= self.max_distance_to_path,
            "inside_queue_extent": self.min_distance_to_counter
            <= feature.distance_to_counter
            <= self.max_distance_to_counter,
            "spacing_consistent": not self.require_neighbor or feature.local_density > 0,
        }
        score = sum(checks.values()) / len(checks)
        reasons = tuple(name for name, passed in checks.items() if passed)
        failures = tuple(f"not_{name}" for name, passed in checks.items() if not passed)
        return RuleResult(all(checks.values()), score, reasons + failures)
