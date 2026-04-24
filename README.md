# Self-Pruning Neural Network 
 
> A neural network that learns to prune its own weights *during* training.

---

## Overview

Standard neural network pruning is a **post-training** step — you train a large
network, then manually remove small weights. This project does something smarter:
the network **learns which weights are unnecessary during training itself**, using
learnable gate parameters and an L1 sparsity regularisation loss.

---

## How It Works

```
Normal Linear:     output = input @ weight.T  +  bias

Prunable Linear:   gates  = sigmoid(gate_scores)          ← learned per-weight scalars
                   output = input @ (weight × gates).T  +  bias
```

If a gate is driven to **0**, the corresponding weight is effectively **removed**.
The L1 penalty on gate values provides a constant gradient pushing every gate toward
zero — unless the classification loss fights back to keep it alive.

```
Total Loss = CrossEntropy(predictions, labels)  +  λ × Σ gate_values
```

---

## Project Structure

```
.
├── Self_Pruning.ipynb   ← main script (all code in one file)
├── results_table.md          ← written report with theory + results table
├── gate_distribution.png     ← generated after training
├── training_curves.png       ← generated after training
└── README.md
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install torch torchvision matplotlib numpy
```

### 2. Run the experiment

```bash
python self_pruning_network.py
```

This will:
- Download CIFAR-10 automatically to `./data/`
- Train 3 separate models with λ = `[0.0001, 0.001, 0.01]`
- Print per-epoch results to console
- Save `gate_distribution.png` and `training_curves.png`
- Save final results in `results_table.md`

### 3. Expected runtime

| Hardware  | Approx. time (30 epochs × 3 runs) |
|:----------|:----------------------------------|
| CPU only  | ~60–90 minutes                    |
| GPU (T4)  | ~8–12 minutes                     |

> Use [Google Colab](https://colab.research.google.com) (free GPU) to run faster.


## Key Files Explained

### `PrunableLinear` 

```python
class PrunableLinear(nn.Module):
    def forward(self, x):
        gates         = torch.sigmoid(self.gate_scores)  # (0, 1)
        pruned_weight = self.weight * gates               # element-wise
        return F.linear(x, pruned_weight, self.bias)
```

- `weight` and `gate_scores` are both `nn.Parameter` — both get gradients.
- No external libraries needed; pure PyTorch autograd handles everything.

### Sparsity Loss

```python
# In SelfPruningNet:
def total_sparsity_loss(self):
    return sum(layer.sparsity_loss() for layer in self.prunable_layers)

# In training loop:
loss = cross_entropy_loss + lambda_sparse * model.total_sparsity_loss()
```

### Sparsity Level Measurement

```python
def sparsity_level(self, threshold=1e-2):
    all_gates = torch.cat([layer.get_gates().view(-1) for layer in self.prunable_layers])
    return (all_gates < threshold).float().mean().item() * 100.0
```

---

## Why L1 Encourages Sparsity 

L1 has a **constant gradient** (unlike L2 which shrinks near zero). This means the
optimiser never "gives up" pushing a gate toward zero — it keeps pushing with equal
force all the way to 0. L2 would only produce small values, not true zeros.

---

## Requirements

```
torch>=1.12
torchvision>=0.13
matplotlib>=3.5
numpy>=1.21
```

---

*Submitted for Tredence Analytics AI Engineering Intern – 2025 Cohort*
