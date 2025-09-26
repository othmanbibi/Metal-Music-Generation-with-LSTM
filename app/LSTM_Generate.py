# generate_midi.py - OPTIMIZED FOR METAL

import torch
import numpy as np
import random
import pretty_midi
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn as nn

# -------------------------
# OPTIMIZED METAL CONFIGURATION
# -------------------------
NPY_DIR = "C:/Projects/Music_ML_Pr/music-ml-app/data/midi/metal/processed/arrays"
MODEL_PATH = "C:/Projects/Music_ML_Pr/music-ml-app/checkpoints/metal_lstm_epoch6.pt"
OUTPUT_DIR = "C:/Projects/Music_ML_Pr/music-ml-app/outputs"

# CORE GENERATION PARAMETERS (CORRECTED)
NUM_RIFFS = 10
LENGTH = 128                # Moderate length for complete phrases
TEMPERATURE = 0.5          # Slightly looser for musicality
PITCH_LOW = 36              # Extended range for drop tunings (E1-C7)
PITCH_HIGH = 84             # Full metal range
STEPS_PER_BEAT = 4          # Back to 16th notes (32nds were too fast)
TEMPO = 90                 # Moderate metal tempo

# CHANNEL-SPECIFIC SETTINGS (REBALANCED)
channel_temp = [0.5, 0.5, 0.5, 0.5]      # Less tight overall
top_k = [5, 5, 5, 5]                       # More options
velocity_map = [90, 80, 85, 70]            # MUCH quieter drums!

# METAL-SPECIFIC PARAMETERS
PALM_MUTE_PROB = 0.8        # 60% palm muted notes for guitar
POWER_CHORD_PROB = 0.2      # 40% power chords
SYNCOPATION_WEIGHT = 1.3    # Emphasize off-beats
MIN_REST_DURATION = 0.5    # Very short rests for tight metal

# -------------------------
# Enhanced Generation Function for Metal
# -------------------------
def generate_metal(model, seed_roll, length=128, top_k_list=None, channel_temp=None):
    model.eval()
    device = next(model.parameters()).device
    generated = seed_roll.clone().to(device)
    hidden = None
    
    C = len(channel_temp) if channel_temp is not None else 4
    P = generated.shape[2] // C
    
    for step in range(length):
        out, hidden = model(generated[:, -1:, :], hidden)
        out = out.squeeze(1)  # shape: (1, C*P)

        new_step = []
        for c in range(C):
            start = c * P
            end = (c + 1) * P
            prob = out[:, start:end]

            # Apply channel-specific temperature
            temp = channel_temp[c] if channel_temp is not None else 1.0
            prob = prob ** (1 / temp)
            
            # Metal-specific adjustments per channel
            if c == 0:  # Guitar channel - favor metal patterns
                prob = apply_guitar_bias(prob, step)
            elif c == 1:  # Bass channel - follow root patterns  
                prob = apply_bass_bias(prob, step)
            elif c == 2:  # Drum channel - emphasize metal beats
                prob = apply_drum_bias(prob, step)

            # Channel-specific top-k
            k = min(top_k_list[c] if top_k_list else top_k, P)
            topk_probs, topk_indices = torch.topk(prob, k)
            topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)
            sampled_idx = torch.multinomial(topk_probs, 1).squeeze(-1)

            note = torch.zeros_like(prob)
            note[0, topk_indices[0, sampled_idx]] = 1.0
            new_step.append(note)

        new_step = torch.cat(new_step, dim=1).unsqueeze(1)
        generated = torch.cat([generated, new_step], dim=1)

    return generated

def apply_guitar_bias(prob, step):
    """Apply metal guitar-specific biases"""
    # Favor power chord intervals (perfect 5th = 7 semitones)
    power_chord_boost = 1.2
    for i in range(len(prob[0]) - 7):
        if prob[0, i] > 0.3:  # If root note likely
            prob[0, i + 7] *= power_chord_boost  # Boost perfect 5th
    
    # Emphasize downbeats for palm muting
    if step % 8 == 0:  # Downbeat
        prob *= 1.1
    
    return prob

def apply_bass_bias(prob, step):
    """Apply metal bass-specific biases"""
    # Favor lower registers and root movements
    low_register_boost = 1.15
    prob[0, :len(prob[0])//3] *= low_register_boost  # Boost lower third
    
    return prob

def apply_drum_bias(prob, step):
    """Apply metal drum-specific biases - REDUCED INTENSITY"""
    # Standard metal kit mapping
    KICK = 8   # Relative position in drum range
    SNARE = 10
    HIHAT = 14
    
    # GENTLER emphasis on kick downbeats
    if step % 4 == 0:  # Every quarter note, not 8th
        if KICK < len(prob[0]):
            prob[0, KICK] *= 1.15  # Reduced from 1.4
    
    # GENTLER emphasis on snare backbeats  
    if step % 4 == 2:  # Beat 3 (backbeat)
        if SNARE < len(prob[0]):
            prob[0, SNARE] *= 1.1  # Reduced from 1.3
    
    # Reduce overall drum density
    prob *= 0.7  # Make drums less likely overall
    
    return prob

# -------------------------
# Enhanced MIDI Conversion for Metal
# -------------------------
def piano_roll_to_metal_midi(roll, pitch_low=28, steps_per_beat=8, tempo=145):
    pm = pretty_midi.PrettyMIDI()
    
    # Metal-appropriate program assignments
    program_map = [
        29,  # Overdriven Guitar (more metal than clean)
        34,  # Electric Bass (finger)
        0,   # Standard Kit
        81   # Lead Synth (for other parts)
    ]
    
    drum_channel = 2
    C, T, P = roll.shape

    for c in range(C):
        is_drum = (c == drum_channel)
        program = 0 if is_drum else program_map[c]
        inst = pretty_midi.Instrument(program=program, is_drum=is_drum)
        vel_base = velocity_map[c]

        for t in range(T):
            for p in range(P):
                if roll[c, t, p] > 0.5:
                    start = t * (60 / tempo) / steps_per_beat
                    
                    # Variable note lengths for metal articulation
                    if c == 0 and random.random() < PALM_MUTE_PROB:
                        # Shorter notes for palm muting
                        duration = (60 / tempo) / steps_per_beat * 0.3
                    else:
                        # Standard note length
                        duration = (60 / tempo) / steps_per_beat * 0.8
                    
                    end = start + duration
                    pitch = pitch_low + p
                    
                    # Channel-specific velocity variation - REBALANCED
                    if c == 0:  # Guitar - moderate variation
                        velocity = int(vel_base * np.random.uniform(0.85, 1.1))
                    elif c == 2:  # Drums - MUCH less variation for consistency
                        velocity = int(vel_base * np.random.uniform(0.9, 1.0))  # Quieter range
                    else:
                        velocity = int(vel_base * np.random.uniform(0.9, 1.1))
                    
                    velocity = max(1, min(127, velocity))
                    note = pretty_midi.Note(
                        velocity=velocity, pitch=int(pitch), start=start, end=end
                    )
                    inst.notes.append(note)
        pm.instruments.append(inst)

    return pm

# -------------------------
# Model and Dataset (unchanged)
# -------------------------
class MetalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=2, dropout=0.2):
        super(MetalLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out, hidden

class MidiPianorollDataset(Dataset):
    def __init__(self, npy_dir: str):
        self.npy_dir = Path(npy_dir)
        self.files = list(self.npy_dir.glob("*.npy"))
        if not self.files:
            raise ValueError(f"No .npy files found in {npy_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        C, T, P = arr.shape
        x = arr.transpose(1, 0, 2).reshape(T, C * P).astype(np.float32)
        return torch.from_numpy(x[:-1, :]), torch.from_numpy(x[1:, :])

def reshape_generated_roll(generated, C, P):
    T = generated.shape[0]
    roll = generated.reshape(T, C, P).permute(1, 0, 2)  # (C, T, P)
    return roll

# -------------------------
# Main Execution
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

dataset = MidiPianorollDataset(NPY_DIR)
sample_input, _ = dataset[0]
input_size = sample_input.shape[1]

# Infer C and P from first file
arr = np.load(dataset.files[0])
C, _, P = arr.shape

# Load model
model = MetalLSTM(input_size=input_size).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

# -------------------------
# Generate Metal Riffs
# -------------------------
print(f"Generating {NUM_RIFFS} metal riffs with optimized parameters...")
print(f"Length: {LENGTH}, Tempo: {TEMPO}, Temperature: {TEMPERATURE}")
print(f"Channel temps: {channel_temp}")
print(f"Velocity map: {velocity_map}")

for i in range(NUM_RIFFS):
    seed_idx = random.randint(0, len(dataset) - 1)
    seed_input, _ = dataset[seed_idx]
    seed_input = seed_input.unsqueeze(0).to(device)

    generated = generate_metal(
        model, 
        seed_input, 
        length=LENGTH, 
        top_k_list=top_k, 
        channel_temp=channel_temp
    )
    
    generated_roll = reshape_generated_roll(generated.squeeze(0), C=C, P=P)
    pm = piano_roll_to_metal_midi(
        generated_roll,
        pitch_low=PITCH_LOW,
        steps_per_beat=STEPS_PER_BEAT,
        tempo=TEMPO,
    )
    
    out_file = output_dir / f"generated_riff_{i+1:03d}.mid"
    pm.write(str(out_file))
    print(f"Saved: {out_file}")

print("Metal riff generation complete!")