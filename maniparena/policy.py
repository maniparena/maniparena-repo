"""Base policy class — participants subclass this in examples/my_policy.py."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ModelPolicy(ABC):
    """Pipeline: convert_input -> run_inference -> convert_output."""

    def __init__(
        self,
        checkpoint_path: str,
        control_mode: str,
        action_horizon: int,
        device: str = "cuda:0",
    ):
        self.checkpoint_path = checkpoint_path
        self.control_mode = control_mode
        self.action_horizon = action_horizon
        self.device = device

        logger.info(f"Loading model from {checkpoint_path}...")
        self.model = self.load_model(checkpoint_path, device)
        logger.info("Model loaded successfully")

    # ── Methods to implement ──────────────────────────────────────

    @abstractmethod
    def load_model(self, checkpoint_path: str, device: str) -> Any:
        """Load your model/checkpoint and return a callable object."""
        ...

    @abstractmethod
    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        """Run forward pass; return raw model output."""
        ...

    @abstractmethod
    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        """Convert raw output to ``{follow1_pos: List[List], follow2_pos: List[List]}``."""
        ...

    # ── Optional overrides ────────────────────────────────────────

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw observation into model input (default: pass-through)."""
        return obs

    def reset(self):
        """Reset stateful policy between episodes."""
        if hasattr(self.model, "reset"):
            self.model.reset()

    # ── Internal ──────────────────────────────────────────────────

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Full inference pipeline (called by the server)."""
        model_input = self.convert_input(obs)
        model_output = self.run_inference(model_input)
        return self.convert_output(model_output)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata sent to client on connect."""
        return {
            "control_mode": self.control_mode,
            "action_horizon": self.action_horizon,
            "state_dim": 14,
            "state_dim_per_arm": 7,
            "protocol_version": "2.0",
        }
