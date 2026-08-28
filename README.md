# Handwritten Digit Recognizer Using CNN

A locally running handwritten digit recognition system built with Python and TensorFlow. Draw or upload a digit and the CNN predicts it instantly — no API, no internet connection required.

---

## Overview

This project trains a Convolutional Neural Network on the MNIST dataset to recognize handwritten digits from 0 to 9. A Streamlit web interface lets you draw a digit on a canvas or upload an image and get an instant prediction with confidence score.

---

## Objective

To build a complete end-to-end machine learning pipeline that:
- Trains a CNN from scratch on real data
- Saves and reloads the trained model
- Preprocesses user-provided images to match the training format
- Delivers predictions through a clean local web interface

---

## Features

- CNN trained on 60,000 MNIST images achieving **99.04% test accuracy**
- Draw a digit directly on the canvas in the browser
- Upload any handwritten digit image (PNG, JPG, BMP)
- Displays predicted digit and confidence percentage
- Shows probability distribution across all 10 digit classes
- Fully local — no API key, no internet required for prediction
- Automated test suite with 14 tests

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.12 | Core language |
| TensorFlow 2.21 / Keras | CNN model building and training |
| NumPy | Array operations |
| Matplotlib | Training plots and confusion matrix |
| scikit-learn | Classification report and confusion matrix |
| Pillow | Image loading and preprocessing |
| Streamlit | Web interface |
| streamlit-drawable-canvas | Drawing canvas in browser |
| pytest | Automated testing |

---

## Dataset

**MNIST** (Modified National Institute of Standards and Technology)

- 70,000 grayscale images of handwritten digits
- 60,000 training images, 10,000 test images
- Each image is 28×28 pixels
- 10 classes: digits 0 through 9
- Pixel values normalized from 0–255 to 0.0–1.0
- Loaded directly via `tensorflow.keras.datasets.mnist` — no manual download needed

---

## CNN Architecture

```
Input (28, 28, 1)
        ↓
Conv2D — 32 filters, 3×3 kernel, ReLU
        ↓
MaxPooling2D — 2×2
        ↓
Conv2D — 64 filters, 3×3 kernel, ReLU
        ↓
MaxPooling2D — 2×2
        ↓
Flatten
        ↓
Dense — 128 units, ReLU
        ↓
Dropout — 50%
        ↓
Dense — 10 units, Softmax
        ↓
Output: probabilities for digits 0–9
```

- **Loss:** sparse_categorical_crossentropy
- **Optimizer:** Adam
- **Metric:** Accuracy
- **Callbacks:** EarlyStopping (patience=3), ModelCheckpoint (saves best weights)

---

## Workflow

```
MNIST Dataset
      ↓
Data Preprocessing (normalize, reshape)
      ↓
CNN Model (Conv → Pool → Conv → Pool → Dense → Dropout → Softmax)
      ↓
Model Training (up to 15 epochs, early stopping)
      ↓
Model Evaluation (accuracy, loss, confusion matrix)
      ↓
Save Trained Model → model/digit_model.keras
      ↓
User Draws or Uploads Digit
      ↓
Image Preprocessing (grayscale → invert → crop → pad → resize → normalize)
      ↓
CNN Prediction
      ↓
Predicted Digit + Confidence Score
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

**Train the model**
```bash
python train.py
```
Trains the CNN, saves the model to `model/digit_model.keras`, and saves training plots to `screenshots/`.

**Evaluate the model**
```bash
python evaluate.py
```
Prints test accuracy, classification report, and saves the confusion matrix to `screenshots/`.

**Run the web app**
```bash
streamlit run app.py
```
Opens the browser at `http://localhost:8501`. Draw or upload a digit and click Predict.

**Run tests**
```bash
python -m pytest tests/test_prediction.py -v
```

---

## Results

All results are from the actual trained model — nothing is fabricated.

| Metric | Value |
|---|---|
| Test Accuracy | 99.04% |
| Test Loss | 0.0293 |
| Training stopped at | Epoch 8 (EarlyStopping) |

Per-class F1-scores:

| Digit | F1-Score |
|---|---|
| 0 | 0.99 |
| 1 | 1.00 |
| 2 | 0.99 |
| 3 | 0.99 |
| 4 | 0.99 |
| 5 | 0.99 |
| 6 | 0.99 |
| 7 | 0.99 |
| 8 | 0.99 |
| 9 | 0.98 |

---

## Screenshots

| | |
|---|---|
| App UI | *(add screenshot)* |
| Training History | *(add screenshot)* |
| Confusion Matrix | *(add screenshot)* |

---

## Project Structure

```
handwritten-digit-recognizer/
│
├── app.py              # Streamlit web interface
├── train.py            # CNN definition and training script
├── predict.py          # Load model and run inference
├── preprocess.py       # MNIST loader + user image preprocessing
├── evaluate.py         # Evaluation metrics and plots
├── visualize.py        # MNIST sample visualization
├── requirements.txt    # Python dependencies
├── .gitignore
├── README.md
│
├── model/
│   └── digit_model.keras   # Saved trained model
│
├── data/
│   └── README.md           # MNIST is loaded automatically
│
├── screenshots/
│   ├── mnist_samples.png
│   ├── training_history.png
│   └── confusion_matrix.png
│
└── tests/
    └── test_prediction.py  # 14 automated tests
```

---

## Future Improvements

- Data augmentation (rotation, shift) to improve robustness on real handwriting
- Batch normalization for faster and more stable training
- Export model to TensorFlow Lite for mobile deployment
- Add support for multi-digit recognition
- Deploy to Streamlit Cloud for public access

---

## License

MIT License — free to use, modify, and distribute.
