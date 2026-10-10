# ML Learning Notes

Small experiments from coursework and self-study. They are kept here as honest learning notes, not as finished projects.

## 1. SVM image classification (spiders vs scorpions)

`svm-spider-scorpion/svm_classification.ipynb`

- Data: Kaggle `mathewribeiro/spiderscorpioncochroach`, 300 images per class, resized to 224×224, augmented with Albumentations.
- Compares a linear SVM with a non-linear (RBF) SVM on 600 test images.

| Model | Test accuracy |
|---|---|
| Linear SVM | 80.67% |
| RBF SVM | 89.33% |

Caveat: the notebook augments the images before the train/val/test split, so augmented copies of one source image can appear on both sides of the split and the accuracies are probably optimistic.

## 2. DGCNN point-cloud classification on ModelNet40

`dgcnn-modelnet40/dgcnn_modelnet40.ipynb`

- Data: ModelNet40 `.off` meshes, sampled to 1,024 points per shape with Open3D.
- Model: DGCNN. Trained for **one epoch only**: train accuracy 57.7%, test accuracy 67.1%.
- A DGCNN + PointNet++ ensemble section is **not finished**.

## Not included

Datasets and trained weights. Paths in the notebooks are Colab-specific (`/content/drive/MyDrive/...`).
