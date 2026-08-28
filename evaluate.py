import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import load_and_preprocess


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(10),
        yticks=np.arange(10),
        xticklabels=range(10),
        yticklabels=range(10),
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix",
    )

    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig("screenshots/confusion_matrix.png", dpi=100)
    plt.show()
    print("Confusion matrix saved to screenshots/confusion_matrix.png")


if __name__ == "__main__":
    model_path = "model/digit_model.keras"
    if not os.path.exists(model_path):
        print("Model not found. Run train.py first.")
        exit(1)

    model = load_model(model_path)
    _, (x_test, y_test) = load_and_preprocess()

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy : {test_acc * 100:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)
