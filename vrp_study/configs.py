from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class ModelConfig:
    # максимальное время работы модели
    max_execution_time_minutes: float = field(default=1)
    # максимальное число решения при работе модели
    max_solution_number: int = field(default=-1)


@dataclass
class ConstraintConfig:
    constraints: List
