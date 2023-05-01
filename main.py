import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from einops import rearrange

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

np.random.seed(0)
torch.manual_seed(0)


# define the ViT class
def patchify(images, n_patches):
    n, c, h, w = images.shape
    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches**2, h * w * c // n_patches**2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d)))
                if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result


class MHSA(nn.Module):
    def __init__(self, n_heads, token_dim, val_dim):
        """_summary_

        Args:
            n_heads (int): the number of multi-heads
            token_dim (int): the dimension of input tokens
            val_dim (int): the dimension of projected QKV
        """
        super(MHSA, self).__init__()
        self.val_dim = int(token_dim / n_heads) if val_dim is None else val_dim
        self.n_heads = n_heads
        self.token_dim = token_dim
        self.scale_factor = self.val_dim ** (-0.5)

        self.to_qkv = nn.Linear(token_dim, 3 * val_dim * n_heads, bias=False)
        self.Merge = nn.Linear(n_heads * val_dim, token_dim, bias=False)

    def forward(self, x):
        assert x.dim() == 3, "Input tensor must be 3D: batch_size, n_patches, token_dim"

        # step 1: linear projection to Q, K, V, output shape = [batch size, n_patches, n_heads*val_dim*3]
        qkv = self.to_qkv(x)

        # step2:  decomposition to q,v,k and cast to tuple, output shape before tuple is[3, batch size, n_heads, n_patches, val_dim]
        q, k, v = tuple(rearrange(qkv, "b p (h v k) -> k b h p v", k=3, h=self.n_heads))

        # step3: calculate attention weights and compute the weighted values, [batch size, n_heads, n_patches, val_dim]
        attention_weights = (
            torch.einsum("b h i d,b h j d -> bhij", q, k) * self.scale_factor
        )
        attention_score = torch.softmax(attention_weights, dim=-1)
        out = torch.einsum("b h i j,b h j d -> b h i d", attention_score, v)

        # step4: merge each head and project to output
        out = rearrange(out, "b h p v -> b p (h v)")
        out = self.Merge(out)

        return out


class MHSA_block(nn.Module):
    def __init__(
        self, n_heads, token_dim, val_dim, MLP_dim, dropout_p, *args, **kwargs
    ) -> None:
        """_summary_

        Args:
            n_heads (int): the number of multi-heads
            token_dim (int): the dimension of input tokens
            val_dim (int): the dimension of projected QKV
            MLP_dim (int): dimension of MLP
            dropout_p (float): dropout probability
        """
        super(MHSA_block, self).__init__(*args, **kwargs)

        self.norm1 = nn.LayerNorm(token_dim)
        self.mhsa = MHSA(n_heads, token_dim, val_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.MLP_layers = nn.Sequential(
            nn.Linear(token_dim, MLP_dim),
            nn.GELU(),
            nn.Linear(MLP_dim, token_dim),
        )

    def forward(self, x):
        o = x + self.mhsa(self.norm1(x))
        o = o + self.MLP_layers(self.norm2(o))

        return o


class MyViT(nn.Module):
    def __init__(
        self,
        chw=(1, 28, 28),
        n_patches=7,
        hidden_d=8,
        n_blocks=1,
        n_heads=1,
        val_dim=3,
        MLP_dim=16,
        dropout_ratio=1.0,
        num_classes=10,
    ) -> None:
        """_summary_

        Args:
            chw (tuple, optional): image shape: (channels, row, col). Defaults to (1, 28, 28).
            n_patches (int, optional): patch_size. Defaults to 7.
            hidden_d (int, optional): token dim. Defaults to 8.
            n_blocks (int, optional): number of multi-head self-attention (MHSA) blocks. Defaults to 1.
            n_heads (int, optional): number of heads in each MHSA. Defaults to 1.
            val_dim (int, optional): QKV dim. Defaults to 3.
            MLP_dim (int, optional): MLP dim in MHSA block. Defaults to 16.
            dropout_ratio (float, optional): dropout probability. Defaults to 1.0.
            num_classes (int, optional): number of classes. Defaults to 10.
        """
        super(MyViT, self).__init__()

        # attribute
        self.chw = chw  # (C, H, W)
        self.n_patches = n_patches
        self.hidden_d = hidden_d
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.val_dim = val_dim
        self.MLP_dim = MLP_dim
        self.dropout_rate = dropout_ratio
        self.num_classes = num_classes

        assert (
            chw[1] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        assert (
            chw[2] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d, bias=False)

        # 2) add learnable classification token
        self.class_token = nn.Parameter(torch.randn(1, self.hidden_d))

        # 3) Positional embedding
        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(n_patches**2 + 1, hidden_d),
            persistent=False,
        )

        # 4. transformer block
        self.MHSA_block = nn.ModuleList(
            [
                MHSA_block(
                    n_heads=self.n_heads,
                    token_dim=self.hidden_d,
                    val_dim=self.val_dim,
                    MLP_dim=self.MLP_dim,
                    dropout_p=self.dropout_rate,
                )
                for _ in range(self.n_blocks)
            ]
        )

        # 5. classification block
        self.class_mlp = nn.Sequential(
            nn.Linear(self.hidden_d, self.num_classes), nn.Softmax(dim=-1)
        )

    def forward(self, images):
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
        tokens = self.linear_mapper(patches)

        # adding classification token
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding

        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # MHSA block
        for block in self.MHSA_block:
            out = block(out)

        # classification block
        out = self.class_mlp(out[:, 0])

        return out


def main():
    # loading data
    transform = ToTensor()

    train_set = MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining model and training options
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
    )

    model = MyViT(
        chw=(1, 28, 28),
        n_patches=7,
        hidden_d=8,
        n_blocks=2,
        n_heads=2,
        val_dim=4,
        MLP_dim=8*4,
        dropout_ratio=1.0,
        num_classes=10,
    ).to(device)

    N_EPOCHS = 5
    LR = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
        ):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == "__main__":
    main()
