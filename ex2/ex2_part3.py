import torch
import torch.nn as nn
from tqdm import tqdm

from ds_downloader import get_dataloaders
from ex2_part1 import Autoencoder, train_autoencoder
from ex2_part2 import ClassifierMLP, train_classifier

# ==========================================
# 1. Configuration
# ==========================================
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 64
NUM_EPOCHS_PRETRAIN = 10
NUM_EPOCHS_CLASSIFIER = 20
LEARNING_RATE = 0.001

LATENT_DIM = 16
CHANNELS = 16

# ==========================================
# 2. Execution Logic
# ==========================================
def run_part3():
    train_loader_full, test_loader, train_loader_subset = get_dataloaders(BATCH_SIZE, subset_size=100)

    print("\n" + "=" * 50)
    print("Part 3 - Step A: Pre-training the Autoencoder")
    print("=" * 50)
    autoencoder = Autoencoder(LATENT_DIM, CHANNELS).to(DEVICE)
    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    ae_loss_fn = nn.L1Loss()

    # Pass title_prefix so Q3 plots are labeled correctly
    train_autoencoder(autoencoder, train_loader_full, test_loader, ae_loss_fn,
                      ae_optimizer, DEVICE, num_epochs=NUM_EPOCHS_PRETRAIN,
                      d=LATENT_DIM, c=CHANNELS, title_prefix="Part 3 (Pre-training)")

    pretrained_encoder = autoencoder.encoder

    for param in pretrained_encoder.parameters():
        param.requires_grad = False
    pretrained_encoder.eval()

    cls_loss_fn = nn.CrossEntropyLoss()

    print("\n" + "=" * 50)
    print("Part 3 - Step C: Training Final MLP on Full Dataset")
    print("=" * 50)
    mlp_full = ClassifierMLP(LATENT_DIM).to(DEVICE)
    optimizer_full = torch.optim.Adam(mlp_full.parameters(), lr=LEARNING_RATE)

    # Added "Part 3" to scenario name for plot title
    train_classifier(pretrained_encoder, mlp_full, train_loader_full, test_loader,
                     cls_loss_fn, optimizer_full, DEVICE, NUM_EPOCHS_CLASSIFIER,
                     "Part 3 - Frozen Encoder + Full Dataset")

    print("\n" + "=" * 50)
    print("Part 3 - Step D: Training Final MLP on 100-Sample Subset")
    print("=" * 50)
    mlp_sub = ClassifierMLP(LATENT_DIM).to(DEVICE)
    optimizer_sub = torch.optim.Adam(mlp_sub.parameters(), lr=LEARNING_RATE)

    # Added "Part 3" to scenario name for plot title
    train_classifier(pretrained_encoder, mlp_sub, train_loader_subset, test_loader,
                     cls_loss_fn, optimizer_sub, DEVICE, NUM_EPOCHS_CLASSIFIER,
                     "Part 3 - Frozen Encoder + 100-Sample Subset")

if __name__ == "__main__":
    run_part3()