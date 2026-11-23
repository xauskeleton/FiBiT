# FiBiT: Extreme Compression of Vision Transformers via Fisher-Guided Pruning and Hybrid Binarization

FiBiT compresses a Vision Transformer in three stages:

1. Fisher-guided structural pruning.
2. Hybrid binarization.
3. DPLR-guided quantization-aware training (QAT).

On CIFAR-100, Swin-S reaches 81.23% top-1 accuracy with 2-bit activations, in 5.1 MB instead of 195.8 MB.

## Setup

Install `torch` and `torchvision` for your CUDA version first, then:

```bash
pip install -r requirements.txt
```

## Run

```bash
bash scripts/run_main.sh                                  # Table 1, Swin-S
MODEL=deit_small_patch16_224 bash scripts/run_main.sh     # DeiT-S
MODEL=vit_small_patch16_224  bash scripts/run_main.sh     # ViT-S

PART=prune     bash scripts/run_ablations.sh              # Table 2
PART=unpruned  bash scripts/run_ablations.sh              # Table 3
PART=partition bash scripts/run_ablations.sh              # Table 4
```

The results go to `ckpt_<backbone>/`. Each run directory has:

- `train.log`: the training log.
- `best.pth`: the best checkpoint.
- `best_packed.pt`: the packed binary model, which is the file size reported in the paper.
