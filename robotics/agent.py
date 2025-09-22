# agent.py — Physical AI Robotics Navigation Agent (fully runnable)
# Interprets camera frames with a TRT-optimized visual encoder + vLLM LLM.
# Falls back gracefully to CPU/stub when CUDA or vLLM is not available.

import time
import numpy as np
import torch
import torch.nn as nn
from enum import Enum
from typing import Optional


class Action(Enum):
    FORWARD   = "FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT= "TURN_RIGHT"
    STOP      = "STOP"
    UNKNOWN   = "UNKNOWN"


class ActionParser:
    """Maps LLM text output to discrete robot actions."""
    KEYWORDS = {
        Action.FORWARD:    ["forward", "ahead", "straight", "go"],
        Action.TURN_LEFT:  ["left", "turn left"],
        Action.TURN_RIGHT: ["right", "turn right"],
        Action.STOP:       ["stop", "halt", "obstacle", "blocked"],
    }

    @classmethod
    def parse(cls, text: str) -> Action:
        text = text.lower()
        for action, keywords in cls.KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return action
        return Action.UNKNOWN


class LightweightVisionEncoder(nn.Module):
    """
    Minimal ViT-style encoder for testing without CLIP/TRT dependencies.
    Produces [B, num_patches, hidden_dim] patch features from raw frames.
    """
    def __init__(self, img_size=224, patch_size=16, hidden_dim=512):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim   = 3 * patch_size * patch_size
        self.patch_size  = patch_size
        self.num_patches = num_patches
        self.proj = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)

    def patchify(self, x):
        """[B,3,H,W] -> [B, num_patches, patch_dim]"""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H//p, p, W//p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C*p*p)
        return x

    def forward(self, x):
        patches = self.patchify(x)
        return self.proj(patches) + self.pos_embed


class LLMDecisionMaker:
    """
    Wraps an LLM (or stub) to produce navigation decisions from visual context.
    Uses a simple transformer decoder stub when a full LLM is unavailable.
    """
    PROMPTS = {
        "navigation": (
            "You are a robot navigation assistant. "
            "Given the visual scene description, output exactly one action: "
            "FORWARD, TURN_LEFT, TURN_RIGHT, or STOP.\n"
            "Scene: {context}\nInstruction: {instruction}\nAction:"
        )
    }

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self.llm = None

        if model_name:
            try:
                from vllm import LLM, SamplingParams
                self.llm = LLM(model=model_name, dtype="float16")
                self.sampling = SamplingParams(temperature=0.1, max_tokens=16)
                print(f"[LLM] Loaded via vLLM: {model_name}")
            except Exception as e:
                print(f"[LLM] vLLM unavailable ({e}), using stub")

        if self.llm is None:
            # Rule-based stub for testing
            self._stub_responses = [
                "FORWARD", "FORWARD", "TURN_LEFT", "FORWARD",
                "TURN_RIGHT", "FORWARD", "STOP", "FORWARD",
            ]
            self._step = 0
            print("[LLM] Using rule-based stub (vLLM not available)")

    def decide(self, visual_context: str, instruction: str) -> str:
        prompt = self.PROMPTS["navigation"].format(
            context=visual_context, instruction=instruction)

        if self.llm is not None:
            outputs = self.llm.generate([prompt], self.sampling)
            return outputs[0].outputs[0].text.strip()
        else:
            # Cycle through stub responses
            resp = self._stub_responses[self._step % len(self._stub_responses)]
            self._step += 1
            return resp


class RoboticsAgent:
    """
    Full Physical AI navigation agent:
      Frame -> Visual Encoder -> LLM Decision Maker -> Action -> Robot Controller
    """
    def __init__(self,
                 llm_model: Optional[str] = None,
                 use_trt: bool = False,
                 device: str = "auto"):

        self.device = ("cuda" if torch.cuda.is_available() else "cpu") \
                      if device == "auto" else device

        # Visual encoder
        if use_trt:
            try:
                from visual_encoder import TRTVisualEncoder
                self.encoder = TRTVisualEncoder()
                print("[Agent] Using TRT visual encoder")
            except Exception:
                self.encoder = LightweightVisionEncoder().to(self.device)
                print("[Agent] TRT unavailable — using lightweight ViT encoder")
        else:
            self.encoder = LightweightVisionEncoder().to(self.device)
        self.encoder.eval()

        self.llm   = LLMDecisionMaker(model_name=llm_model, device=self.device)
        self.stats = {"steps": 0, "total_latency_ms": 0.0}
        print(f"[Agent] Ready on {self.device} | TRT={use_trt}")

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """HxWxC uint8 numpy -> 1x3x224x224 float tensor."""
        from PIL import Image
        img = Image.fromarray(frame).resize((224, 224))
        t   = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        return ((t - mean) / std).unsqueeze(0).to(self.device)

    def describe_scene(self, patch_features: torch.Tensor) -> str:
        """Convert patch features to a text description (stub: uses feature stats)."""
        mean_act = patch_features.mean().item()
        if mean_act > 0.1:
            return "open corridor with clear path ahead"
        elif mean_act > 0:
            return "room with possible obstacle on the left"
        else:
            return "blocked path, obstacle directly ahead"

    def decide(self, frame: np.ndarray, instruction: str) -> Action:
        t0 = time.perf_counter()

        tensor = self.preprocess_frame(frame)
        with torch.no_grad():
            patch_feats = self.encoder(tensor)

        scene_desc  = self.describe_scene(patch_feats)
        action_text = self.llm.decide(scene_desc, instruction)
        action      = ActionParser.parse(action_text)

        lat_ms = (time.perf_counter() - t0) * 1000
        self.stats["steps"] += 1
        self.stats["total_latency_ms"] += lat_ms
        return action

    def run_loop(self, num_steps: int = 10,
                 instruction: str = "Navigate to the exit door"):
        print(f"\n[Agent] Starting control loop: '{instruction}'")
        print(f"{'Step':>5} {'Action':>12} {'Latency':>10}")
        print("-" * 32)

        for step in range(num_steps):
            # Simulate camera frame (random noise in real test)
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            action = self.decide(frame, instruction)
            avg_lat = self.stats["total_latency_ms"] / self.stats["steps"]
            print(f"{step+1:>5} {action.value:>12} {avg_lat:>8.1f}ms")

            if action == Action.STOP:
                print("[Agent] STOP action — halting loop.")
                break

        print(f"\n[Agent] Done. Avg latency: "
              f"{self.stats['total_latency_ms']/self.stats['steps']:.1f}ms")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",       type=int, default=10)
    parser.add_argument("--llm-model",   default=None)
    parser.add_argument("--use-trt",     action="store_true")
    parser.add_argument("--instruction", default="Navigate to the exit door")
    args = parser.parse_args()

    agent = RoboticsAgent(llm_model=args.llm_model, use_trt=args.use_trt)
    agent.run_loop(num_steps=args.steps, instruction=args.instruction)
