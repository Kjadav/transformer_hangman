"""
This script trains a character-level Transformer model for the Hangman game using masked language modeling (MLM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import math


# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('positional_encode', pe)

    def forward(self, x):
        x = x + self.positional_encode[:, :x.size(1), :]
        return x



# -------------------------
# Transformer Model
# -------------------------
class HangmanTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.output(x)


# -------------------------
# Vocabulary Helper
# -------------------------
class Vocab:
    def __init__(self, words):
        characters = set("".join(words))
        characters.discard('_')
        characters = sorted(characters)
        self.character_index = {char: idx + 1 for idx, char in enumerate(characters)}
        self.character_index['_'] = 0
        self.index_character = {idx: char for char, idx in self.character_index.items()}

    def encode(self, word):
        return [self.character_index[char] for char in word]


# -------------------------
# Dataset for MLM
# -------------------------
class HangmanDataset(Dataset):
    def __init__(self, words, vocab, max_len=50, mask_prob=0.25):
        self.data = []
        self.vocab = vocab
        self.max_len = max_len

        for word in words:
            word = word.strip().lower()
            if not word or len(word) > max_len:
                continue

            encoded = vocab.encode(word)
            input_ids = encoded.copy()
            labels = [-1] * len(encoded)

            for i in range(len(encoded)):
                if random.random() < mask_prob:
                    labels[i] = input_ids[i]
                    input_ids[i] = 0  # Mask with underscore

            # Padding
            pad_len = max_len - len(encoded)
            input_ids += [0] * pad_len
            labels += [-1] * pad_len

            self.data.append((input_ids, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, targets = self.data[idx]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


# -------------------------
# Masking and Loss
# -------------------------
def mask_creation(input_tensor, token_id=0):
    return (input_tensor != token_id)


def masked_cross_entropy_loss(logits, targets, ignore_index=-1):
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    mask = targets != ignore_index
    return F.cross_entropy(logits[mask], targets[mask])


# -------------------------
# Training Loop
# -------------------------
def train_model(model, dataloader, optimizer, device, epochs=10):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            mask = ~mask_creation(inputs)

            outputs = model(inputs, mask)
            loss = masked_cross_entropy_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")


# -------------------------
# Optional: Evaluation
# -------------------------
def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            mask = ~mask_creation(inputs)

            outputs = model(inputs, mask)
            loss = masked_cross_entropy_loss(outputs, targets)
            total_loss += loss.item()

    print(f"Validation Loss: {total_loss / len(dataloader):.4f}")


# -------------------------
# Main Training Entry Point
# -------------------------
if __name__ == "__main__":
    # Load and preprocess words
    with open("words_250000_train.txt") as f:
        words = [line.strip().lower() for line in f if line.strip()]
        random.seed(42)
        words = random.sample(words, 100000)

    # Build vocab and dataset
    vocab = Vocab(words)
    vocab_size = max(vocab.character_index.values()) + 1
    dataset = HangmanDataset(words, vocab, max_len=50, mask_prob=0.25)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model setup
    model = HangmanTransformer(vocab_size, d_model=64, nhead=4, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    train_model(model, dataloader, optimizer, device, epochs=10)

    # Save model
    torch.save(model.state_dict(), "hangman_transformer.pt")
    print("Training complete. Model saved.")
