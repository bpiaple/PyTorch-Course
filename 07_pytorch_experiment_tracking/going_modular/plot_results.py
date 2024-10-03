from typing import Dict, List

import matplotlib.pyplot as plt

# Plot the results

def plot_results(results: Dict[str, List[float]]) -> None:
    """Plots the results of a PyTorch model.

    Args:
        results (Dict[str, List[float]]): A dictionary containing the results.
    """
    # Get the loss values of the results dictionary (train and test)
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get the accuracy values of the results dictionary (train and test)
    train_accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Create a range of the number of epochs
    epochs = range(len(train_loss))


    # Create a figure and axis
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Plot the loss values
    axs[0].plot(epochs, train_loss, label="Train Loss", color="blue", marker="o")
    axs[0].plot(epochs, test_loss, label="Test Loss", color="red", marker="o")
    axs[0].set_title("Loss vs. Number of Epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot the accuracy values
    axs[1].plot(epochs, train_accuracy, label="Train Accuracy", color="blue", marker="o")
    axs[1].plot(epochs, test_accuracy, label="Test Accuracy", color="red", marker="o")
    axs[1].set_title("Accuracy vs. Number of Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    