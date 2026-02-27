# AI vs Real Image Detector

An image classification system using **EfficientNetB4** that detects whether an image is **AI-Generated (FAKE)** or **Real**, with **Grad-CAM heatmap** visualization showing which regions of the image influenced the model's decision.

---

## Project Structure

```
ai-image-detector/
├── train_model.py       # Model training script (2-phase transfer learning)
├── gradcam_utils.py     # Grad-CAM heatmap generation utilities
├── app.py               # Streamlit web application
├── requirements.txt     # Python dependencies
├── README.md            # This file
│
├── dataset/             # CIFAKE dataset (you must provide this)
│   ├── train/
│   │   ├── FAKE/        # 50,000 AI-generated images
│   │   └── REAL/        # 50,000 real images
│   └── test/
│       ├── FAKE/        # 10,000 AI-generated images
│       └── REAL/        # 10,000 real images
│
├── model/               # Created after training
│   ├── efficientnetb4_ai_detector.keras
│   └── model_meta.json
│
└── plots/               # Created after training
    ├── accuracy.png
    └── loss.png
```

---

## Prerequisites

- **Python 3.10 – 3.13**
- **pip** (Python package manager)
- The **CIFAKE dataset** (download from [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images))

---

## Setup Instructions

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
| Package        | Purpose                            |
|----------------|------------------------------------|
| tensorflow     | Deep learning framework            |
| keras          | High-level neural network API      |
| opencv-python  | Image processing & Grad-CAM overlay|
| numpy          | Numerical operations               |
| Pillow         | Image loading                      |
| matplotlib     | Training plots                     |
| streamlit      | Web UI                             |
| scikit-learn   | Classification metrics             |

### Step 2: Prepare the Dataset

1. Download the CIFAKE dataset from Kaggle
2. Extract it so the folder structure looks like:
   ```
   dataset/
   ├── train/
   │   ├── FAKE/    (50,000 images)
   │   └── REAL/    (50,000 images)
   └── test/
       ├── FAKE/    (10,000 images)
       └── REAL/    (10,000 images)
   ```
3. The `dataset/` folder must be in the same directory as the Python files.

### Step 3: Train the Model

**Quick test** (verifies pipeline works, ~5 min on CPU):
```bash
python train_model.py --quick
```

**Full training** (high accuracy, ~2-4 hours on CPU, ~30 min on GPU):
```bash
python train_model.py
```

#### What training does:
1. **Phase 1** — Trains only the classifier head (EfficientNetB4 base frozen)
2. **Phase 2** — Fine-tunes the top 30 layers of EfficientNetB4
3. Evaluates on the test set and prints accuracy, classification report, confusion matrix
4. Saves the model to `model/efficientnetb4_ai_detector.keras`
5. Saves training plots to `plots/accuracy.png` and `plots/loss.png`

### Step 4: Launch the Web App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## How to Use the Web App

1. **Upload** any image (JPG, PNG, or WEBP)
2. The model classifies it as **AI-Generated** or **Real**
3. A **confidence score** shows how sure the model is
4. **Grad-CAM heatmap** shows which regions influenced the decision:
   - 🔴 **Red** = high attention (strong influence)
   - 🟡 **Yellow** = medium attention
   - 🟢 **Green** = low attention

### Sidebar Options
- **Heatmap opacity slider** — adjust overlay transparency
- **JET colormap toggle** — switch from R/Y/G to standard JET colors

---

## Technical Details

| Component      | Detail                                          |
|----------------|-------------------------------------------------|
| Base Model     | EfficientNetB4 (ImageNet pretrained)             |
| Input Size     | 380×380 (full) / 160×160 (quick mode)           |
| Output         | Sigmoid — binary classification                  |
| Training       | 2-phase: frozen base → fine-tune top 30 layers   |
| Augmentation   | Random flip, rotation, zoom                      |
| Optimizer      | Adam with ReduceLROnPlateau                      |
| Grad-CAM       | Last conv layer of EfficientNet backbone         |
| Framework      | TensorFlow 2.18+ / Keras 3.x                    |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model file not found` | Run `python train_model.py` first |
| `Port 8501 already in use` | Kill old process or use `streamlit run app.py --server.port 8502` |
| `Out of memory during training` | Reduce `BATCH_SIZE` in `train_model.py` (try 8 or 4) |
| `Slow training on CPU` | Use `--quick` flag or use a GPU machine |

---

## License

This project is for educational purposes. The CIFAKE dataset has its own licensing terms on Kaggle.
