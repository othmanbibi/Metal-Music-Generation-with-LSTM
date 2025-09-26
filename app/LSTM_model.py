import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os





class MidiPianorollDataset(Dataset):
    """
    PyTorch Dataset for MIDI piano-roll slices.
    Each slice: [C, T, P], where
    C = channels (guitar, bass, drums, other)
    T = time steps
    P = pitch range
    """
    def __init__(self, npy_dir: str):
        self.npy_dir = Path(npy_dir)
        self.files = list(self.npy_dir.glob("*.npy"))
        if not self.files:
            raise ValueError(f"No .npy files found in {npy_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])  # [C, T, P], uint8

        C, T, P = arr.shape
        # Flatten channels × pitches → feature vector per time step
        x = arr.transpose(1, 0, 2).reshape(T, C*P).astype(np.float32)

        # Input: all time steps except last
        input_seq = x[:-1, :]
        # Target: next-step prediction
        target_seq = x[1:, :]

        return torch.from_numpy(input_seq), torch.from_numpy(target_seq)


dataset = MidiPianorollDataset(npy_dir="data/midi/metal/processed/arrays")
batch_size = 32

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)



class MetalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=2, dropout=0.2):
        super(MetalLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()  # binary output for piano-roll

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out, hidden
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample_input, _ = dataset[0]
input_size = sample_input.shape[1] 
model = MetalLSTM(input_size=input_size).to(device)
criterion = nn.BCELoss()  # binary cross-entropy
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50

# Directory to save checkpoints
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

num_epochs = 50
print_every = 64 

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Print loss every 64 batches
        if (batch_idx) % print_every == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")

    # Save checkpoint per epoch
    checkpoint_path = os.path.join(checkpoint_dir, f"metal_lstm_epoch{epoch+1}.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item()
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}\n")

