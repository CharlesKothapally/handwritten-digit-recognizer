import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.datasets import mnist


def prepare_image(image: Image.Image) -> np.ndarray:
    """Convert a user-provided PIL image into CNN-ready format."""
    # Convert to grayscale
    image = image.convert("L")

    # Invert if background is white (MNIST has white digit on black background)
    pixels = np.array(image)
    if pixels.mean() > 127:
        image = ImageOps.invert(image)

    # Crop tightly around the digit content (removes empty border)
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    # Add padding so digit is not touching edges (matches MNIST style)
    image = ImageOps.expand(image, border=20, fill=0)

    # Resize to 28x28
    image = image.resize((28, 28), Image.LANCZOS)

    # Normalize to 0-1
    arr = np.array(image).astype("float32") / 255.0

    # Reshape to (1, 28, 28, 1) for CNN
    return arr.reshape(1, 28, 28, 1)


def load_and_preprocess():
    """Load MNIST and return preprocessed train/test splits."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape: (samples, 28, 28) -> (samples, 28, 28, 1) for CNN input
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_and_preprocess()
    print(f"Training samples : {x_train.shape[0]}")
    print(f"Testing  samples : {x_test.shape[0]}")
    print(f"Image shape      : {x_train.shape[1:]}")
    print(f"Pixel range      : {x_train.min():.1f} – {x_train.max():.1f}")
    print(f"Label sample     : {y_train[:10]}")
