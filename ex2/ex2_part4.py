import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from ds_downloader import get_dataloaders
from ex2_part1 import Autoencoder, Decoder, train_autoencoder
from ex2_part2 import Encoder, ClassifierMLP, train_classifier

# ==========================================
# 1. Configuration
# ==========================================
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 0.001

LATENT_DIM = 16
CHANNELS = 16


# ==========================================
# 2. Hybrid Model Definition
# ==========================================
class HybridAutoencoder(nn.Module):
    def __init__(self, frozen_encoder, trainable_decoder):
        super().__init__()
        self.encoder = frozen_encoder
        self.decoder = trainable_decoder

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)


# ==========================================
# 3. Comparison Visualization
# ==========================================
def compare_reconstructions(model_q1, model_q4, loader, device, num_images=10):
    model_q1.eval()
    model_q4.eval()

    with torch.no_grad():
        images, labels = next(iter(loader))
        images = images[:num_images].to(device)

        recon_q1 = model_q1(images)
        recon_q4 = model_q4(images)

        fig, axes = plt.subplots(3, num_images, figsize=(15, 6))
        # Updated Title to show Part 4.a-d
        fig.suptitle("Part 4 (a-d) - Comparison: Q1 (Unsupervised) vs Q4 (Task-Specific Classification) Encoding",
                     fontsize=16)

        for i in range(num_images):
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title(f"Label: {labels[i].item()}")
            axes[0, i].axis('off')

            axes[1, i].imshow(recon_q1[i].cpu().squeeze(), cmap='gray')
            if i == 0: axes[1, i].set_title("Q1 Recon", fontweight='bold')
            axes[1, i].axis('off')

            axes[2, i].imshow(recon_q4[i].cpu().squeeze(), cmap='gray')
            if i == 0: axes[2, i].set_title("Q4 Recon", fontweight='bold')
            axes[2, i].axis('off')

        plt.tight_layout()
        plt.show()


def compare_reconstructions_specific_digit(model_q1, model_q4, loader, device, target_digit=2, num_images=10):
    """Gathers specific digits from the loader and plots their reconstructions."""
    model_q1.eval()
    model_q4.eval()

    # Collect specific digits
    target_images = []
    target_labels = []
    for images, labels in loader:
        for i in range(len(labels)):
            if labels[i].item() == target_digit:
                target_images.append(images[i])
                target_labels.append(labels[i])
            if len(target_images) == num_images:
                break
        if len(target_images) == num_images:
            break

    # Stack the list of tensors into a single batch tensor
    images = torch.stack(target_images).to(device)

    with torch.no_grad():
        recon_q1 = model_q1(images)
        recon_q4 = model_q4(images)

        fig, axes = plt.subplots(3, num_images, figsize=(15, 6))
        fig.suptitle(f"Part 4 (c) - In-Class Variability: Q1 vs Q4 Encoding (Digit {target_digit})", fontsize=16)

        for i in range(num_images):
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title(f"Label: {target_labels[i].item()}")
            axes[0, i].axis('off')

            axes[1, i].imshow(recon_q1[i].cpu().squeeze(), cmap='gray')
            if i == 0: axes[1, i].set_title("Q1 Recon", fontweight='bold')
            axes[1, i].axis('off')

            axes[2, i].imshow(recon_q4[i].cpu().squeeze(), cmap='gray')
            if i == 0: axes[2, i].set_title("Q4 Recon", fontweight='bold')
            axes[2, i].axis('off')

        plt.tight_layout()
        plt.show()


# ==========================================
# 4. Execution Logic
# ==========================================
def run_part4():
    train_loader_full, test_loader, _ = get_dataloaders(BATCH_SIZE)

    print("\n" + "=" * 50)
    print("Part 4 - STEP 1: Training Q1 Autoencoder (Reconstruction)")
    print("=" * 50)
    model_q1 = Autoencoder(LATENT_DIM, CHANNELS).to(DEVICE)
    opt_q1 = torch.optim.Adam(model_q1.parameters(), lr=LEARNING_RATE)
    loss_l1 = nn.L1Loss()

    train_autoencoder(model_q1, train_loader_full, test_loader, loss_l1, opt_q1, DEVICE, NUM_EPOCHS, LATENT_DIM,
                      CHANNELS, title_prefix="Part 4 (Step 1)")

    print("\n" + "=" * 50)
    print("Part 4 - STEP 2: Training Q2 Encoder for Classification")
    print("=" * 50)
    encoder_q2 = Encoder(LATENT_DIM, CHANNELS).to(DEVICE)
    mlp_q2 = ClassifierMLP(LATENT_DIM).to(DEVICE)
    opt_q2 = torch.optim.Adam(list(encoder_q2.parameters()) + list(mlp_q2.parameters()), lr=LEARNING_RATE)
    loss_ce = nn.CrossEntropyLoss()

    train_classifier(encoder_q2, mlp_q2, train_loader_full, test_loader, loss_ce, opt_q2, DEVICE, NUM_EPOCHS,
                     "Part 4 (Step 2) - Q2 Classifier")

    print("\n" + "=" * 50)
    print("Part 4 - STEP 3: Training Q4 Decoder on Frozen Q2 Encoder")
    print("=" * 50)
    for param in encoder_q2.parameters():
        param.requires_grad = False
    encoder_q2.eval()

    decoder_q4 = Decoder(LATENT_DIM, CHANNELS).to(DEVICE)
    model_q4 = HybridAutoencoder(encoder_q2, decoder_q4).to(DEVICE)
    opt_q4 = torch.optim.Adam(decoder_q4.parameters(), lr=LEARNING_RATE)

    train_autoencoder(model_q4, train_loader_full, test_loader, loss_l1, opt_q4, DEVICE, NUM_EPOCHS, LATENT_DIM,
                      CHANNELS, title_prefix="Part 4 (Step 3 - Hybrid)")

    print("\nGenerating visual comparison...")
    compare_reconstructions(model_q1, model_q4, test_loader, DEVICE)

    print("\nGenerating specific digit comparison (digit 2)...")
    compare_reconstructions_specific_digit(model_q1, model_q4, test_loader, DEVICE, target_digit=2)


if __name__ == "__main__":
    run_part4()