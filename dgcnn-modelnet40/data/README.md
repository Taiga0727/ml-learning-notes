# Data

## Source

**ModelNet40** (Princeton), downloaded by the notebook from Kaggle: `balraj98/modelnet40-princeton-3d-object-dataset` (about 1.9 GB zip; the Kaggle page lists the licence as "other"). The official copy is also at <https://modelnet.cs.princeton.edu/ModelNet40.zip>. The shapes are **not redistributed here**.

| Fact | Value |
|---|---|
| Shapes | 12,311 `.off` meshes in 40 categories |
| Official split | 9,843 train, 2,468 test (`metadata_modelnet40.csv` lists every shape with class, split and path) |
| Size | about 9.1 GiB extracted |
| Largest shapes | `curtain/train/curtain_0130.off` 165 MiB, `airplane/train/airplane_0050.off` 112 MiB: they dominate loading time and memory |

## The `.off` header quirk

The first line of the official files is `OFF6360 10283 0`: the word `OFF` and the vertex, face and edge counts share one line. Open3D (and many other readers) expect `OFF` on its own line. The notebook's `fix_off_header` rewrites each file in place as `OFF`, newline, `6360 10283 0`, which adds exactly one byte per file. Keep this in mind when comparing a processed copy against the original download.

## Derived data (not stored here)

`ModelNet40_Augmented` holds `.npy` point clouds (1,024 points each) generated from the **training** shapes only: the original sampling plus two augmented copies, 29,529 files in total. The augmentation is a random rotation around the z-axis, a random scale between 0.8 and 1.25, and an optional random point dropout of up to 10% (dropped points are replaced by duplicates). This set was **not** used for the DGCNN run in this repository.
