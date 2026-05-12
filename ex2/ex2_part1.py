import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from ds_downloader import get_dataloaders

# ==========================================
# 1. Configuration & Constants
# ==========================================
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

LATENT_DIMS = [4, 16]
CHANNEL_COUNTS = [4, 16]

# ==========================================
# 2. Architecture Definitions
# ==========================================
class Encoder(nn.Module):
    def __init__(self, latent_dim, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(channels * 2 * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, channels):
        super().__init__()
        self.channels = channels
        self.fc = nn.Linear(latent_dim, channels * 2 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.channels * 2, 7, 7)
        return self.deconv(x)

class Autoencoder(nn.Module):
    def __init__(self, latent_dim, channels):
        super().__init__()
        self.encoder = Encoder(latent_dim, channels)
        self.decoder = Decoder(latent_dim, channels)

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

# ==========================================
# 3. Training & Visualization Functions
# ==========================================
def train_autoencoder(model, data_loader, test_loader, loss_function,
                      optimizer, device, num_epochs, d, c, title_prefix="Part 1.b/c"):
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        curr_train_loss = []

        for data_input, _ in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            data_input = data_input.to(device)
            prediction = model(data_input)
            loss = loss_function(prediction, data_input)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            curr_train_loss.append(loss.item())

        avg_train_loss = sum(curr_train_loss) / len(curr_train_loss)
        train_losses.append(avg_train_loss)

        model.eval()
        curr_test_loss = []
        with torch.no_grad():
            for data_input, _ in test_loader:
                data_input = data_input.to(device)
                prediction = model(data_input)
                loss = loss_function(prediction, data_input)
                curr_test_loss.append(loss.item())

        avg_test_loss = sum(curr_test_loss) / len(curr_test_loss)
        test_losses.append(avg_test_loss)

        print(f"Epoch: [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    # Updated Plot Title
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.xlabel("epoch")
    plt.ylabel("loss (Mean L1)")
    plt.title(f"{title_prefix} - Reconstruction Loss (d={d}, channels={c})")
    plt.legend()
    plt.show()

    return train_losses, test_losses


def visualize_reconstruction(model, loader, device, d, c, title_prefix="Part 1.b/c"):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(loader))
        images = images[:8].to(device)
        outputs = model(images)

        fig, axes = plt.subplots(2, 8, figsize=(15, 4))
        # Updated Plot Title
        fig.suptitle(f"{title_prefix} - Reconstructions (d={d}, channels={c})", fontsize=16)
        for i in range(8):
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
            axes[0, i].set_title("Original")
            axes[0, i].axis('off')
            axes[1, i].imshow(outputs[i].cpu().squeeze(), cmap='gray')
            axes[1, i].set_title("Recon")
            axes[1, i].axis('off')
        plt.show()

# ==========================================
# 4. Execution Logic
# ==========================================
def run_part1():
    train_loader, test_loader, _ = get_dataloaders(BATCH_SIZE)

    for channels in CHANNEL_COUNTS:
        for d in LATENT_DIMS:
            print(f"\n{'=' * 50}")
            print(f"Part 1: Training Config: Latent Dim (d) = {d}, Channels = {channels}")
            print(f"{'=' * 50}")

            model = Autoencoder(latent_dim=d, channels=channels).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            loss_fn = nn.L1Loss()

            train_autoencoder(model, train_loader, test_loader, loss_fn,
                              optimizer, DEVICE, NUM_EPOCHS, d, channels, title_prefix="Part 1")

            visualize_reconstruction(model, test_loader, DEVICE, d, channels, title_prefix="Part 1")

if __name__ == "__main__":
    run_part1()