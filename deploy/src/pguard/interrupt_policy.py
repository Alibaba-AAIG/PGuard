"""Stage-scoped interrupt policy evaluated on guard scores already on the CPU."""

from dataclasses import dataclass
import json
import math
from typing import NamedTuple

import numpy as np


STAGES = ("prefill", "decode")


class NonFiniteGuardScores(ValueError):
    pass


@dataclass(frozen=True)
class InterruptHit:
    rule: str
    probability: float
    threshold: float

    @property
    def message(self):
        return (
            f"pguard: {self.rule}; probability={self.probability:.8g}; "
            f"threshold={self.threshold:.8g}"
        )


class GuardScoreRows(list):
    """Keeps the response list format and also holds the CPU array copied from the device."""

    def __init__(self, rows, array):
        super().__init__(rows)
        self.array = array


class HeadRules(NamedTuple):
    start: int
    stop: int
    rules: tuple


class _Plan(NamedTuple):
    targets: np.ndarray
    thresholds: np.ndarray
    rules: tuple


def is_unscored(rows):
    return all(isinstance(row, (int, float)) and row == 0.0 for row in rows)


def load_policy(rules, head_config_path):
    with open(head_config_path, encoding="utf-8") as handle:
        return InterruptPolicy(rules, json.load(handle))


def _head_bounds(config):
    if not isinstance(config, dict) or config.get("label_type", "single") != "single":
        raise ValueError("interrupt policy requires a single-label classifier config")
    classifier = config.get("classifier")
    heads = classifier.get("classification_heads") if isinstance(classifier, dict) else None
    schema = config.get("task_schema")
    if not isinstance(heads, (list, tuple)) or not heads or not isinstance(schema, dict):
        raise ValueError("interrupt policy requires classification_heads and task_schema")
    bounds = {}
    width = 0
    for task_id, entry in enumerate(heads):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(f"invalid classification head: {entry!r}")
        name, size = entry
        if (not isinstance(name, str) or not name or name in bounds
                or isinstance(size, bool) or not isinstance(size, int) or size <= 0):
            raise ValueError(f"invalid classification head: {entry!r}")
        task = schema.get(name)
        if not isinstance(task, dict) or task.get("task_id") != task_id:
            raise ValueError(f"task_schema.{name} does not match the classification head order")
        label2id, id2label = task.get("label2id"), task.get("id2label")
        if not isinstance(label2id, dict) or not isinstance(id2label, dict):
            raise ValueError(f"task_schema.{name} is missing label mappings")
        if (any(not isinstance(label, str) or isinstance(index, bool)
                or not isinstance(index, int) for label, index in label2id.items())
                or set(label2id.values()) != set(range(size))
                or len(label2id) != size
                or id2label != {str(index): label for label, index in label2id.items()}):
            raise ValueError(f"task_schema.{name} label mappings are inconsistent or out of range")
        bounds[name] = (width, width + size)
        width += size
    if set(schema) != set(bounds):
        raise ValueError("task_schema does not match classification_heads")
    return schema, bounds, width


def _parse_rules(rules, schema, bounds):
    entries = {stage: {} for stage in STAGES}
    seen = set()
    for text in rules:
        rule = text.strip()
        category, at, threshold_text = rule.rpartition("@")
        stage, dot, category = category.partition(".")
        task, delimiter, label = category.partition("###")
        if not at or not dot or not delimiter or stage not in entries:
            raise ValueError(f"invalid rule, expected stage.category###label@threshold: {text!r}")
        try:
            threshold = float(threshold_text)
        except ValueError as exc:
            raise ValueError(f"invalid threshold: {text!r}") from exc
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"threshold must be a finite number in [0, 1]: {text!r}")
        if task not in bounds or label not in schema[task]["label2id"]:
            raise ValueError(f"unknown classification head or label: {text!r}")
        key = (stage, task, label)
        if key in seen:
            raise ValueError(f"duplicate rule: {text!r}")
        seen.add(key)
        entries[stage].setdefault(task, []).append(
            (schema[task]["label2id"][label], threshold, rule)
        )
    return {
        stage: tuple(HeadRules(*bounds[task], tuple(items)) for task, items in tasks.items())
        for stage, tasks in entries.items()
    }


class InterruptPolicy:
    def __init__(self, rules, config):
        if not isinstance(rules, list) or not all(isinstance(rule, str) for rule in rules):
            raise ValueError("interrupt_config must be a list of strings")
        schema, bounds, self.width = _head_bounds(config)
        self.groups = _parse_rules(rules, schema, bounds)
        self.plans = {stage: self._compile(groups) for stage, groups in self.groups.items()}

    @property
    def enabled(self):
        return any(self.groups.values())

    @staticmethod
    def _compile(groups):
        if not groups:
            return None
        targets, thresholds, rules = [], [], []
        for head in groups:
            for index, threshold, rule in head.rules:
                targets.append(head.start + index)
                thresholds.append(threshold)
                rules.append(rule)
        return _Plan(np.asarray(targets), np.asarray(thresholds), tuple(rules))

    def match(self, rows, stage):
        if stage not in self.groups:
            raise ValueError(f"unknown stage: {stage!r}")
        if not self.groups[stage] or len(rows) == 0:
            return [None] * len(rows)
        if is_unscored(rows):
            return [None] * len(rows)
        if any(not hasattr(row, "__len__") or len(row) != self.width for row in rows):
            raise ValueError("guard_logits width does not match the classification heads")
        array = getattr(rows, "array", None)
        if array is None:
            array = np.asarray(rows, dtype=np.float64)
        if array.shape != (len(rows), self.width):
            raise ValueError("CPU score array does not match the request rows")
        if not np.isfinite(array).all():
            raise NonFiniteGuardScores("guard_logits contains non-finite values")
        plan = self.plans[stage]
        probabilities = array[:, plan.targets]
        matches = probabilities > plan.thresholds
        result = [None] * len(rows)
        for row_id in np.flatnonzero(matches.any(axis=1)):
            rule_id = int(matches[row_id].argmax())
            result[row_id] = InterruptHit(plan.rules[rule_id], float(probabilities[row_id, rule_id]),
                                          float(plan.thresholds[rule_id]))
        return result
