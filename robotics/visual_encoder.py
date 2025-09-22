# visual_encoder.py — TensorRT-Optimized Visual Encoder (fully runnable)
# Runs TRT engine if available; falls back to PyTorch ViT stub automatically.

import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class PyTorchVisualEncoder(nn.Module):
    """Pure PyTorch ViT-style encoder — always works, no TRT dependency."""
    def __init__(self, img_size=224, patch_size=16, hidden_dim=512):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim   = 3 * patch_size * patch_size
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pos  = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8,
                dim_feedforward=hidden_dim*4,
                batch_first=True, dropout=0.0)
            for _ in range(4)
        ])

    def patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H//p, p, W//p, p)
        return x.permute(0,2,4,1,3,5).reshape(B,-1,C*p*p)

    def forward(self, x):
        tokens = self.proj(self.patchify(x)) + self.pos
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)


class TRTVisualEncoder:
    """
    TensorRT-accelerated ViT encoder.
    Loads a serialized .trt engine if it exists; otherwise falls back to PyTorch.
    """
    def __init__(self, engine_path: str = "models/vit_int8.trt",
                 device: str = "cuda", hidden_dim: int = 512):
        self.device     = device
        self.hidden_dim = hidden_dim
        self.engine     = None
        self.context    = None
        self.use_trt    = False

        if Path(engine_path).exists():
            self._load_trt_engine(engine_path)
        else:
            print(f"[TRTEncoder] Engine not found at {engine_path}")
            print("[TRTEncoder] Falling back to PyTorch encoder")

        if not self.use_trt:
            self.pytorch_encoder = PyTorchVisualEncoder(hidden_dim=hidden_dim)
            if device == "cuda" and torch.cuda.is_available():
                self.pytorch_encoder = self.pytorch_encoder.cuda()
            self.pytorch_encoder.eval()

    def _load_trt_engine(self, engine_path: str):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime    = trt.Runtime(TRT_LOGGER)

            with open(engine_path, "rb") as f:
                self.engine  = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

            # Allocate I/O buffers
            self.bindings = []
            self.host_buffers   = []
            self.device_buffers = []
            for binding in self.engine:
                size  = trt.volume(self.engine.get_binding_shape(binding))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                h_buf = cuda.pagelocked_empty(size, dtype)
                d_buf = cuda.mem_alloc(h_buf.nbytes)
                self.bindings.append(int(d_buf))
                self.host_buffers.append(h_buf)
                self.device_buffers.append(d_buf)

            import pycuda.driver as cuda_driver
            self.stream = cuda_driver.Stream()
            self.use_trt = True
            print(f"[TRTEncoder] Loaded TRT engine: {engine_path}")
        except Exception as e:
            print(f"[TRTEncoder] TRT load failed: {e}")

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, 224, 224] float32 tensor
        returns: [B, num_patches, hidden_dim]
        """
        if self.use_trt:
            return self._trt_infer(images)
        else:
            with torch.no_grad():
                return self.pytorch_encoder(
                    images.to(next(self.pytorch_encoder.parameters()).device))

    def _trt_infer(self, images: torch.Tensor) -> torch.Tensor:
        import pycuda.driver as cuda

        np_input = images.cpu().numpy().astype(np.float32)
        np.copyto(self.host_buffers[0], np_input.ravel())
        cuda.memcpy_htod_async(self.device_buffers[0],
                               self.host_buffers[0], self.stream)
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_buffers[1],
                               self.device_buffers[1], self.stream)
        self.stream.synchronize()

        B = images.size(0)
        out = torch.from_numpy(
            self.host_buffers[1].reshape(B, -1, self.hidden_dim).copy())
        return out.to(images.device)

    def benchmark(self, batch_size: int = 1, n: int = 100):
        images = torch.randn(batch_size, 3, 224, 224)
        # Warmup
        for _ in range(10): self.encode(images)
        torch.cuda.synchronize() if self.device == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n): self.encode(images)
        torch.cuda.synchronize() if self.device == "cuda" else None
        ms = (time.perf_counter() - t0) / n * 1000
        mode = "TRT" if self.use_trt else "PyTorch"
        print(f"[{mode}] BS={batch_size} | {ms:.2f}ms/frame")
        return ms


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="models/vit_int8.trt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    enc = TRTVisualEncoder(engine_path=args.engine)

    dummy = torch.randn(args.batch_size, 3, 224, 224)
    out   = enc.encode(dummy)
    print(f"Output shape: {out.shape}  (expected [{args.batch_size}, 196, 512])")

    if args.benchmark:
        enc.benchmark(batch_size=args.batch_size)
