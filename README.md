# Chest X-Ray Pneumonia Classifier

A transfer-learning pneumonia classifier (DenseNet121) with Grad-CAM
explainability, deployed as a Streamlit app. Built as a portfolio project
going beyond "trained a CNN" to include evaluation rigor, interpretability,
and deployment.

> ⚠️ **Not a diagnostic tool.** Student project for educational/portfolio
> purposes only. Not validated for clinical use.

## Problem statement

TODO: 2-3 sentences — what pneumonia screening from chest X-rays is used
for clinically, why a screening-oriented classifier optimizes for recall
over precision, and what this project demonstrates.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
— Kaggle, 5,863 pediatric chest X-rays labeled NORMAL / PNEUMONIA.

> Kermany, D.S. et al. "Identifying Medical Diagnoses and Treatable Diseases
> by Image-Based Deep Learning." Cell, 2018.

## Approach

1. **Baseline CNN** — small 4-conv-block model trained from scratch, as a
   sanity-check comparison point.
2. **Transfer learning** — DenseNet121 (ImageNet weights), custom
   classification head, trained in two phases: frozen base, then fine-tuned
   at a low learning rate with the top layers unfrozen.
3. **Class weighting** to correct for the ~3x class imbalance
   (more PNEUMONIA than NORMAL images in the training set).
4. **Grad-CAM** for interpretability — visualizes which regions of the
   X-ray drove each prediction.

## Results

TODO: fill in after running `evaluate.py`.

| Model              | Accuracy | AUC | Precision (Pneumonia) | Recall (Pneumonia) | F1 |
|---------------------|----------|-----|------------------------|----------------------|----|
| Baseline CNN         | TODO     | TODO| TODO                   | TODO                 | TODO |
| DenseNet121 (frozen) | TODO     | TODO| TODO                   | TODO                 | TODO |
| DenseNet121 (fine-tuned) | TODO | TODO| TODO                   | TODO                 | TODO |

**Why recall matters most here:** in a screening context, a missed pneumonia
case (false negative) is worse than a false alarm (false positive) — a false
alarm gets caught at the next stage of review, a missed case doesn't.
Precision/recall are tracked and reported per class instead of relying on
overall accuracy, which is misleading on an imbalanced dataset.

## Grad-CAM examples

TODO: insert 3-4 images from `outputs/gradcam/` here after running
`python -m src.gradcam`. Include at least one interesting/borderline case
and briefly note whether the heatmap focuses on the lungs (good sign) or on
borders/artifacts (bad sign, worth investigating).

## Project structure

```
chest-xray-classifier/
├── app.py                  # Streamlit deployment app
├── download_data.py        # pulls dataset from Kaggle into data/chest_xray
├── requirements.txt
├── README.md
├── .gitignore
└── src/
    ├── config.py            # all paths & hyperparameters
    ├── data_pipeline.py     # tf.data loading, augmentation, class weights
    ├── model.py              # baseline CNN + DenseNet121 transfer model
    ├── train.py               # training orchestration, both models
    ├── evaluate.py             # test-set metrics, confusion matrix, ROC
    └── gradcam.py               # Grad-CAM heatmap generation
```

## How to run

```bash
pip install -r requirements.txt

# 1. Kaggle API token at ~/.kaggle/kaggle.json, then:
python download_data.py

# 2. Train both models (baseline CNN, then DenseNet121 frozen -> fine-tune)
python -m src.train

# 3. Evaluate on the test set (touched once)
python -m src.evaluate --model outputs/models/densenet121_finetuned.keras

# 4. Generate Grad-CAM examples
python -m src.gradcam --random-samples 6

# 5. Run the app locally
streamlit run app.py
```

## Live demo

TODO: link once deployed to Streamlit Community Cloud or Hugging Face Spaces.

## Limitations & clinical disclaimer

- **Single-institution dataset** — all images come from one pediatric
  population at Guangzhou Women and Children's Medical Center; the model
  has not been validated on adult patients, other institutions, or other
  imaging equipment.
- **No external validation** — performance numbers above reflect a held-out
  split of the same source dataset, not an independent test set.
- **Not for real medical use** — this is a portfolio/learning project. Any
  real pneumonia screening tool would require clinical validation,
  regulatory review, and radiologist oversight before deployment.
- Class imbalance and labeling were handled at the dataset-curator level;
  this project did not independently verify ground-truth labels.

## Acknowledgments

Dataset: Kermany et al., Mendeley Data / Kaggle (see citation above).
