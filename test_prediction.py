import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from PIL import Image, ImageDraw
from preprocess import prepare_image, load_and_preprocess
from predict import load_trained_model, predict


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    return load_trained_model()


def make_digit_image(digit="1", bg="black", fg="white", size=(280, 280)):
    """Create a simple synthetic digit image for testing."""
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    if digit == "1":
        draw.line([size[0]//2, 20, size[0]//2, size[1]-20], fill=fg, width=20)
    elif digit == "0":
        draw.ellipse([40, 20, size[0]-40, size[1]-20], outline=fg, width=20)
    return img


# ── preprocess tests ──────────────────────────────────────────────────────────

def test_prepare_image_output_shape():
    img = make_digit_image()
    arr = prepare_image(img)
    assert arr.shape == (1, 28, 28, 1), f"Expected (1,28,28,1), got {arr.shape}"


def test_prepare_image_pixel_range():
    img = make_digit_image()
    arr = prepare_image(img)
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_prepare_image_white_background_inverted():
    # White background image should be inverted so digit is bright on dark
    img = make_digit_image(bg="white", fg="black")
    arr = prepare_image(img)
    # After inversion the digit pixels should be bright (> 0)
    assert arr.max() > 0.1, "Inverted image should have bright digit pixels"


def test_prepare_image_non_empty():
    img = make_digit_image()
    arr = prepare_image(img)
    assert arr.sum() > 0, "Preprocessed image should not be all zeros"


# ── MNIST data tests ──────────────────────────────────────────────────────────

def test_mnist_shapes():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess()
    assert x_train.shape == (60000, 28, 28, 1)
    assert x_test.shape  == (10000, 28, 28, 1)
    assert y_train.shape == (60000,)
    assert y_test.shape  == (10000,)


def test_mnist_pixel_range():
    (x_train, _), _ = load_and_preprocess()
    assert x_train.min() >= 0.0
    assert x_train.max() <= 1.0


def test_mnist_label_range():
    (_, y_train), (_, y_test) = load_and_preprocess()
    assert y_train.min() == 0 and y_train.max() == 9
    assert y_test.min()  == 0 and y_test.max()  == 9


# ── Model loading tests ───────────────────────────────────────────────────────

def test_model_loads(model):
    assert model is not None


def test_model_raises_if_missing():
    from predict import load_trained_model
    import unittest.mock as mock
    with mock.patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            load_trained_model()


# ── Prediction tests ──────────────────────────────────────────────────────────

def test_predict_returns_correct_types(model):
    img = make_digit_image()
    digit, confidence, probs = predict(img, model)
    assert isinstance(digit, int)
    assert isinstance(confidence, float)
    assert isinstance(probs, np.ndarray)


def test_predict_digit_in_range(model):
    img = make_digit_image()
    digit, _, _ = predict(img, model)
    assert 0 <= digit <= 9


def test_predict_confidence_in_range(model):
    img = make_digit_image()
    _, confidence, _ = predict(img, model)
    assert 0.0 <= confidence <= 100.0


def test_predict_probabilities_sum_to_one(model):
    img = make_digit_image()
    _, _, probs = predict(img, model)
    assert len(probs) == 10
    assert abs(probs.sum() - 1.0) < 1e-5


def test_predict_on_mnist_sample(model):
    # Feed MNIST test image directly through the model (bypassing prepare_image)
    # to confirm the model itself is correct on clean MNIST data
    _, (x_test, y_test) = load_and_preprocess()
    probs = model.predict(x_test[:1], verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(probs[digit]) * 100
    assert digit == int(y_test[0]), f"Expected {y_test[0]}, got {digit}"
    assert confidence > 50.0
