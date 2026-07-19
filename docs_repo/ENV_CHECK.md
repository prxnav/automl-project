# Environment Check

`pip freeze | md5sum` must match on all three machines (docs/06 §6.2).
Also verify torch sees the GPU: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
⚠️ RTX 50-series (Blackwell) may need a recent CUDA wheel — discover on day 1, not day 4.

| Machine | Member | pip freeze md5 | torch sees GPU | Date |
|---|---|---|---|---|
| rtx5070-laptop | A (Karan) | | | |
| rtx5070-desktop | B (Sridhar) | | | |
| rtx3060 | C (Menon) | | | |
