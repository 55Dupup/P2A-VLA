from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ModalityConfig:
    delta_indices: List[int]
    modality_keys: List[str]


class BasePolicy(ABC):
    @abstractmethod
    def step(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        raise NotImplementedError
