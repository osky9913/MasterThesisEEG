import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

from src.dataset import DASPSDataset
from src.model import EEGModelLSTM

input_size = 924  # Number of EEG features
hidden_size = 128  # Number of units in the LSTM hidden layer
num_layers = 2  # Number of LSTM layers
num_classes = 2  # Number of anxiety classes
EEG_PATH = "../datasets/Dasps.mat/"
LABEL_PATH = "../datasets/participant_rating_public.xlsx"
subject_range = range(1, 24)
    mps_device = torch.device("cpu")


if __name__ == "__main__":
    # Define paths and subject range

    train_dataset = DASPSDataset(
        EEG_PATH, LABEL_PATH, range(1, 21), flag_min_rocket=True, device=mps_device
    )
    test_dataset = DASPSDataset(
        EEG_PATH, LABEL_PATH, range(21, 24), flag_min_rocket=True, device=mps_device
    )
    val_dataset = DASPSDataset(
        EEG_PATH, LABEL_PATH, range(21, 24), flag_min_rocket=True, device=mps_device
    )

    # Define data loaders without shuffling for validation and test sets
    batch_size = 2  # Set your desired batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Initialize the LSTM model
    input_size = 336  # Number of EEG features
    hidden_size = 168  # Number of units in the LSTM hidden layer
    num_layers = 8  # Number of LSTM layers
    num_classes = 4  # Number of anxiety classes
    model = EEGModelLSTM(input_size, hidden_size, num_layers, num_classes)
    model.to(device=mps_device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Training - Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()  # Set the model to evaluation mode
    test_total_correct = 0
    test_total_samples = 0

    id = 0
    results = {}
    with torch.no_grad():
        for test_data, test_labels in test_loader:
            test_outputs = model(test_data)
            true_l = test_labels.tolist()
            predict_l = test_outputs.tolist()

            for batch in range(len(true_l)):
                print(true_l[batch])
                print(predict_l[batch])
                results[str(id)] = (true_l[batch], predict_l[batch])
                id += 1

    true_labels = []
    model_outputs = []

    # Extract true labels and model outputs from the dictionary
    for key, value in results.items():
        true_labels.append(value[0])
        model_outputs.append(value[1])

    # Convert the lists to numpy arrays
    true_labels = np.array(true_labels)
    model_outputs = np.array(model_outputs)

    # Calculate ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(true_labels.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], model_outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure()
    for i in range(true_labels.shape[1]):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Multi-Class")
    plt.legend(loc="lower right")
    plt.show()

    plt.figure()
    plt.bar(range(true_labels.shape[1]), roc_auc.values())
    plt.xticks(
        range(true_labels.shape[1]),
        labels=[f"Class {i}" for i in range(true_labels.shape[1])],
    )
    plt.xlabel("Classes")
    plt.ylabel("AUC")
    plt.title("AUC Values for Each Class")
    plt.show()

    print(results)

"""

        # Validation loop
        # Inside the validation loop
        if (epoch + 1) % 100 == 0:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for val_data, val_labels in val_loader:
                    # Forward pass
                    val_outputs = model(val_data)
                    val_loss += criterion(val_outputs, val_labels).item()

                    # Apply sigmoid and convert to binary predictions
                    sigmoid_outputs = torch.sigmoid(val_outputs)
                    predicted = (sigmoid_outputs > 0.5).long()  # Convert to long (int)

                    total += val_labels.size(0)
                    correct += (predicted == val_labels).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = 100.0 * correct / total
            print(
                f"Validation - Epoch [{epoch+1}/{num_epochs}], Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%"
            )

    # Testing loop
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for test_data, test_labels in test_loader:
            # Forward pass
            test_outputs = model(test_data)
            test_loss += criterion(test_outputs, test_labels).item()
            _, predicted = test_outputs.max(1)
            total += test_labels.size(0)
            correct += predicted.eq(test_labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100.0 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

"""
