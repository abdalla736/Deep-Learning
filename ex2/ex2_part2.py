import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from ds_downloader import get_dataloaders
from ex2_part1 import Encoder

# ==========================================
# 1. Configuration
# ==========================================
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

LATENT_DIM = 16
CHANNELS = 16


# ==========================================
# 2. Added MLP Architecture
# ==========================================
class ClassifierMLP(nn.Module):
    def __init__(self, latent_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# ==========================================
# 3. Training & Evaluation Function
# ==========================================
def train_classifier(encoder, mlp, train_loader, test_loader, loss_function,
                     optimizer, device, num_epochs, scenario_name):
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(num_epochs):
        encoder.train()
        mlp.train()
        curr_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for data_input, labels in tqdm(train_loader, desc=f"[{scenario_name}] Epoch {epoch + 1}/{num_epochs}",
                                       leave=False):
            data_input, labels = data_input.to(device), labels.to(device)

            latent = encoder(data_input)
            predictions = mlp(latent)
            loss = loss_function(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curr_train_loss += loss.item() * data_input.size(0)
            _, predicted = torch.max(predictions.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = curr_train_loss / total_train
        train_acc = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        encoder.eval()
        mlp.eval()
        curr_test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for data_input, labels in test_loader:
                data_input, labels = data_input.to(device), labels.to(device)

                latent = encoder(data_input)
                predictions = mlp(latent)
                loss = loss_function(predictions, labels)

                curr_test_loss += loss.item() * data_input.size(0)
                _, predicted = torch.max(predictions.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        avg_test_loss = curr_test_loss / total_test
        test_acc = correct_test / total_test
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f}, Acc: {test_acc:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Title directly uses the scenario_name which we will format from the calling scripts
    fig.suptitle(f"{scenario_name}")

    ax1.plot(train_losses, label="Train Error (Loss)")
    ax1.plot(test_losses, label="Test Error (Loss)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()

    ax2.plot(train_accuracies, label="Train Accuracy")
    ax2.plot(test_accuracies, label="Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.show()

    return train_losses, test_losses, train_accuracies, test_accuracies


# ==========================================
# 4. Execution Logic
# ==========================================
def run_part2():
    train_loader_full, test_loader, train_loader_subset = get_dataloaders(BATCH_SIZE, subset_size=100)
    loss_fn = nn.CrossEntropyLoss()

    print("\n" + "=" * 50)
    print("Part 2 - Scenario (i): Training on Full Dataset")
    print("=" * 50)
    encoder_full = Encoder(LATENT_DIM, CHANNELS).to(DEVICE)
    mlp_full = ClassifierMLP(LATENT_DIM).to(DEVICE)
    optimizer_full = torch.optim.Adam(list(encoder_full.parameters()) + list(mlp_full.parameters()), lr=LEARNING_RATE)

    # Added "Part 2" to scenario name for plot title
    train_classifier(encoder_full, mlp_full, train_loader_full, test_loader, loss_fn,
                     optimizer_full, DEVICE, NUM_EPOCHS, "Part 2 - Full Dataset (60,000)")

    print("\n" + "=" * 50)
    print("Part 2 - Scenario (ii): Training on 100-Sample Subset")
    print("=" * 50)
    encoder_sub = Encoder(LATENT_DIM, CHANNELS).to(DEVICE)
    mlp_sub = ClassifierMLP(LATENT_DIM).to(DEVICE)
    optimizer_sub = torch.optim.Adam(list(encoder_sub.parameters()) + list(mlp_sub.parameters()), lr=LEARNING_RATE)

    # Added "Part 2" to scenario name for plot title
    train_classifier(encoder_sub, mlp_sub, train_loader_subset, test_loader, loss_fn,
                     optimizer_sub, DEVICE, NUM_EPOCHS, "Part 2 - 100-Sample Subset")


if __name__ == "__main__":
    run_part2()