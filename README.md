# PicoTensor

A minimal **pure-Python** Tensor library built from scratch for learning how multidimensional arrays (tensors) actually work under the hood.

This project follows a step-by-step approach: starting with pure 1D, then 2D, and eventually moving up to 3D and 4D. The goal is to deeply understand shapes, broadcasting, memory layout, and tensor operations — the foundation of modern machine learning frameworks like PyTorch and NumPy.

### Why PicoTensor?
- Educational: Every version is kept clean and self-contained so you can clearly see the progression.
- No dependencies — 100% pure Python.
- Designed for learning, not production speed.

## Current Versions

I believe the fastest and most effective way to truly understand how tensor libraries like NumPy and PyTorch work is to build one yourself from scratch. So that's exactly what I'm doing — step by step, version by version.

| Version   | Folder     | Features                                                                                                                | Status            |
|-----------|------------|-------------------------------------------------------------------------------------------------------------------------|-------------------|
| **v1-1d** | `v1-1d/`   | 1D Tensor (vectors)<br>Basic arithmetic (`+`, `-`, `*`, `/`)<br>`sum()`, `mean()`, `max()`, `min()`<br>Indexing         | ✅ Complete       |
| **v2-2d** | `v2-2d/`   | 2D Tensor (matrices)<br>Shape & ndim tracking<br>Matrix multiplication (`@`)<br>Element-wise operations<br>`flatten()`  | ✅ Initial version|
| **v3-3d** | `v3-3d/`   | 3D Tensor support                                                        | Planned           |
| **v4-4d** | `v4-4d/`   | Full 4D Tensor                                                           | Planned           |
