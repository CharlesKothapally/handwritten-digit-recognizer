import matplotlib.pyplot as plt
from preprocess import load_and_preprocess


def show_samples(x, y, n=10):
    """Display n sample images with their labels."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("MNIST Sample Images", fontsize=14)

    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"Label: {y[i]}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("screenshots/mnist_samples.png", dpi=100)
    plt.show()
    print("Sample image saved to screenshots/mnist_samples.png")


if __name__ == "__main__":
    (x_train, y_train), _ = load_and_preprocess()
    show_samples(x_train, y_train)
