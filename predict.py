import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from preprocess import prepare_image

MODEL_PATH = "model/digit_model.keras"


def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Please run train.py first."
        )
    return load_model(MODEL_PATH)


def predict(image: Image.Image, model=None):
    """
    Accept a PIL Image, preprocess it, and return (digit, confidence, probabilities).
    """
    if model is None:
        model = load_trained_model()

    arr = prepare_image(image)
    probs = model.predict(arr, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(probs[digit]) * 100
    return digit, confidence, probs


if __name__ == "__main__":
    from preprocess import load_and_preprocess

    # Quick smoke-test: predict on a real MNIST test image
    model = load_trained_model()
    _, (x_test, y_test) = load_and_preprocess()

    # Convert first test image back to PIL for the pipeline
    sample = (x_test[0].reshape(28, 28) * 255).astype("uint8")
    pil_img = Image.fromarray(sample, mode="L")

    digit, confidence, probs = predict(pil_img, model)
    print(f"True Label       : {y_test[0]}")
    print(f"Predicted Digit  : {digit}")
    print(f"Confidence       : {confidence:.2f}%")
    print(f"All Probabilities: { {i: f'{p*100:.2f}%' for i, p in enumerate(probs)} }")
