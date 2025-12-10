# Multimodal AI Agents
> 🔬 **Research / Exploratory** — End-to-end multimodal systems: a GPU-accelerated VQA model and a Physical AI agent for robotic navigation, both built on TensorRT-optimized vision encoders and efficient LLM serving.

Two projects that sit at the top of the stack — they consume the low-level kernels and inference optimizations from the sibling repos and apply them to real-world AI agent tasks.

---

## Repository Structure

```
multimodal-ai-agents/
├── vqa/         ← Multimodal LLM for Visual Question Answering (BLIVA-inspired)
└── robotics/    ← Physical AI agent for robotic navigation (vLLM + TensorRT)
```

---

## How This Repo Connects to the Stack

```
cuda-kernels-and-primitives        llm-inference-optimization
        │                                      │
        │  AVX2 ops, CUDA activations          │  TRT engines, fused kernels
        └──────────────┬───────────────────────┘
                       ▼
            multimodal-ai-agents
          (VQA model + Robotics agent)
```

---

## Projects

### 🔷 `vqa/` — Multimodal LLM for Visual Question Answering
A BLIVA-inspired multimodal LLM integrating ViT image patch features with a Q-Former bridge and LLaMA-based decoder, fine-tuned on VQA and OCR-heavy datasets using a two-stage training framework.

**Architecture:**
```
Image → ViT Encoder → Q-Former (cross-attn) ─┐
                                               ├─→ LLM Decoder (LoRA) → Answer
Text  → Tokenizer  → Embeddings ──────────────┘
```

**Two-Stage Training:**
| Stage | Frozen | Trainable | Goal |
|---|---|---|---|
| Stage 1 | LLM | Q-Former + Visual Proj | Image-text alignment |
| Stage 2 | Visual Encoder | Q-Former + LLM (LoRA) | VQA instruction tuning |

**Results:**
| Dataset | BLIP-2 Baseline | This Model | Δ |
|---|---|---|---|
| VQAv2 (val) | 65.0% | 67.2% | +2.2% |
| TextVQA | 42.5% | 44.8% | +2.3% |
| OCR-VQA | 38.1% | 41.5% | **+3.4%** |

```bash
pip install torch transformers peft open-clip-torch
python vqa/stage2_finetune.py
python vqa/eval_vqav2.py
```

---

### 🔷 `robotics/` — Physical AI Agent for Robotic Navigation
A multimodal navigation agent combining a TensorRT-optimized visual encoder with vLLM-served LLM for low-latency edge deployment, trained on synthetic data from simulated environments with varied lighting conditions.

**Architecture:**
```
Camera Frame
     │
TRT Visual Encoder (INT8, edge-optimized)
     │  Visual Tokens
Multimodal LLM (vLLM, continuous batching)
     │  Action Text
Action Parser → robot primitives (FORWARD / TURN / STOP)
     │
Robot Controller
```

**Inference Latency:**
| Component | FP32 | FP16 | INT8 (TRT) |
|---|---|---|---|
| Visual Encoder | 28ms | 14ms | **8ms** |
| LLM Decode | 42 tok/s | 81 tok/s | 95 tok/s |
| End-to-end decision | 210ms | 105ms | **78ms** |

```bash
pip install torch vllm transformers
# Start vLLM server
python robotics/vllm_server.py --model llava-hf/llava-1.6-mistral-7b-hf
# Run agent
python robotics/agent.py --camera /dev/video0

# Or via Docker
docker build -t robotics-agent robotics/deployment/
docker run --gpus all robotics-agent
```

---

## Requirements
```
torch >= 2.1
transformers >= 4.38
peft >= 0.9
vllm >= 0.4.0
open-clip-torch >= 2.20
Pillow >= 10.0
CUDA >= 12.0
```
