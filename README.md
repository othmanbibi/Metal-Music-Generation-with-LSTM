# Metal Music Generation with LSTM

A deep learning pipeline for generating metal music using LSTM neural networks. This project processes MIDI files, trains an LSTM model on metal music patterns, and generates new metal riffs and compositions.

## 🎸 Features

- **MIDI Preprocessing**: Converts MIDI files into tempo-invariant piano roll representations
- **Multi-Channel Architecture**: Separates guitar, bass, drums, and other instruments for better learning
- **Metal-Optimized Generation**: Specialized parameters for metal music characteristics (palm muting, power chords, syncopation)
- **High-Quality Audio Output**: Converts generated MIDI to professional audio using FluidSynth and multiple SoundFonts
- **Configurable Pipeline**: Easily adaptable parameters for different metal subgenres

## 🎵 Sample Outputs

Listen to what the model can generate:

### 🎸 Generated Metal Riffs
[![Play Sample Riff 1](https://img.shields.io/badge/▶️-Play%20Riff%201-red)](https://github.com/othmanbibi/Metal-Music-Generation-with-LSTM/raw/main/transformed_riffs/generated-riff-002-metal.mp4)




*Note: Sample files are generated using the trained model and high-quality SoundFonts. Each generation is unique due to the stochastic sampling process.*

## 🏗️ Project Structure

```
Metal-Music-Generation-with-LSTM/
├── app/
    ├── preprocess_midi_pipeline.py    # MIDI preprocessing and piano roll conversion
    ├── LSTM_model.py                  # Model definition and training
    ├── LSTM_Generate.py               # Music generation from trained models  
    └── midi_to_music.py               # MIDI to audio conversion pipeline
├── data/
│   └── midi/metal/
│       ├── raw/                   # Raw MIDI files
│       └── processed/arrays/      # Preprocessed numpy arrays
├── checkpoints/                   # Trained model checkpoints
├── generated_midi_files/          # Generated MIDI files
├── transformed_riffs/             # Final audio outputs
│   ├── generated_riff_001.mp3
│   ├── generated_riff_002.mp3
│   └── ...
└── SoundFont/                     # SoundFont files for realistic audio
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- FluidSynth
- FFmpeg
- Required Python packages (see requirements below)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Metal-Music-Generation-with-LSTM.git
cd Metal-Music-Generation-with-LSTM
```

2. Install Python dependencies:
```bash
pip install torch numpy pandas tqdm pretty_midi pathlib mido
```

3. Install system dependencies:
   - **FluidSynth**: Download from [FluidSynth releases](https://github.com/FluidSynth/fluidsynth/releases)
   - **FFmpeg**: Download from [FFmpeg downloads](https://ffmpeg.org/download.html)

4. Download SoundFonts:
   - Place metal guitar, bass, and drum SoundFonts in the `SoundFont/` directory
   - Update paths in `midi_to_music.py` configuration

### Usage

#### 1. Preprocess MIDI Data

Convert raw MIDI files to training-ready piano roll format:

```bash
python preprocess_midi_pipeline.py \
  --genre metal \
  --raw_dir data/midi/metal/raw \
  --out_dir data/midi/metal/processed \
  --transpose -2 0 2 \
  --bars 4 \
  --hop_bars 2 \
  --steps_per_beat 4 \
  --pitch_low 36 \
  --pitch_high 84 \
  --min_duration_sec 10.0
```

#### 2. Train the Model

Train the LSTM on preprocessed data:

```bash
python LSTM_model.py
```

The model will train for 50 epochs and save checkpoints in `checkpoints/`.

#### 3. Generate Music

Generate new metal riffs using a trained model:

```bash
python LSTM_Generate.py
```

This creates MIDI files in the `outputs/` directory.

#### 4. Convert to Audio

Convert generated MIDI files to high-quality audio:

```bash
python midi_to_music.py
```

Final MP3 files will be saved in the `audio/` directory.

## 🎵 Model Architecture

### LSTM Network
- **Input**: Piano roll representation [Channels × Time × Pitch]
- **Architecture**: 2-layer LSTM with 512 hidden units
- **Output**: Multi-channel binary piano roll predictions
- **Training**: Binary cross-entropy loss with gradient clipping

### Multi-Channel Design
- **Channel 0**: Guitar (distorted, palm-muted patterns)
- **Channel 1**: Bass (low-frequency foundation)
- **Channel 2**: Drums (kick, snare, hi-hat patterns)
- **Channel 3**: Other instruments (leads, harmonies)

### Metal-Specific Features
- **Power Chord Detection**: Emphasizes perfect 5th intervals
- **Palm Muting Simulation**: Shorter note durations for tighter sound
- **Syncopation Weighting**: Emphasizes off-beat patterns
- **Drum Pattern Optimization**: Metal-appropriate kick/snare patterns

## ⚙️ Configuration

### Key Parameters

**Generation Settings:**
```python
NUM_RIFFS = 10           # Number of riffs to generate
LENGTH = 128             # Length in time steps
TEMPERATURE = 0.5        # Sampling temperature
TEMPO = 90              # BPM for output
```

**Metal-Specific:**
```python
PALM_MUTE_PROB = 0.8    # Probability of palm muting
POWER_CHORD_PROB = 0.2  # Probability of power chords
SYNCOPATION_WEIGHT = 1.3 # Off-beat emphasis
```

**Audio Quality:**
```python
SAMPLE_RATE = 44100     # Audio sample rate
MP3_QUALITY = 1         # MP3 encoding quality (0-9)
```

## 📊 Training Results

The model trains on preprocessed metal MIDI data and learns:
- Characteristic metal rhythmic patterns
- Guitar-bass-drum interactions
- Harmonic progressions typical of metal genres
- Timing and articulation patterns

Training typically converges after 20-30 epochs with proper regularization.

## 🎛️ Customization

### Adding New Genres

1. Update `GENRE_CONFIGS` in `preprocess_midi_pipeline.py`
2. Modify channel mappings in `map_instrument_channel()`
3. Adjust generation parameters for genre characteristics

### Improving Audio Quality

1. Use high-quality SoundFonts for each instrument
2. Adjust velocity mappings in `midi_to_music.py`
3. Modify audio processing effects (compression, EQ)

## 🔧 Troubleshooting

**Common Issues:**

1. **No MIDI files found**: Ensure raw MIDI files are in the correct directory
2. **CUDA out of memory**: Reduce batch size or model size
3. **FluidSynth errors**: Check FluidSynth installation and SoundFont paths
4. **Empty audio output**: Verify SoundFont compatibility and MIDI validity

## 📈 Future Improvements

- [ ] Transformer-based architecture for longer-term dependencies
- [ ] Velocity-sensitive generation (continuous rather than binary)
- [ ] Real-time generation and playback
- [ ] Web interface for interactive generation
- [ ] Multi-genre training with style transfer

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement:

- Additional metal subgenre support
- Better audio post-processing
- Performance optimizations
- Documentation improvements

## 🎼 Acknowledgments

- Built using PyTorch deep learning framework
- MIDI processing with pretty_midi library
- Audio synthesis with FluidSynth
- Inspired by advances in AI music generation research

## 📧 Contact

For questions or suggestions, or if you need trained models or the data please open an issue or contact [Othman.BIBI@emines.um6p.ma].

---

**Generated metal riffs are for educational and creative purposes. Rock on! 🤘**
