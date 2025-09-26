import subprocess
from pathlib import Path
from mido import MidiFile, MidiTrack, Message
import random
import logging
import json
from typing import Dict, List, Optional
import os

# -------------------------
# Configuration
# -------------------------
class Config:
    def __init__(self, config_file: Optional[Path] = None):
        # Default configuration
        self.MIDI_DIR = Path("C:/Projects/Music_ML_Pr/music-ml-app/outputs")
        self.OUTPUT_DIR = Path("C:/Projects/Music_ML_Pr/music-ml-app/audio")
        self.FLUIDSYNTH_EXE = Path(r"C:\fluidsynth-2.4.8-win10-x64\bin\fluidsynth.exe")
        
        # SoundFonts per channel (back to multi-SF for best metal realism)
        self.CHANNEL_SOUNDFONTS = {
            0: Path(r"C:\Projects\Music_ML_Pr\SoundFont\Super_Heavy_Guitar_Collection.sf2"),
            1: Path(r"C:\Projects\Music_ML_Pr\SoundFont\sr18_bass.sf2"),
            2: Path(r"C:\Projects\Music_ML_Pr\SoundFont\Drums_TamaRockSTAR.sf2"),
            3: Path(r"C:\Projects\Music_ML_Pr\SoundFont\OmegaGMGS2.sf2"),
        }
        
        # Rebalanced metal-specific audio processing settings
        self.VELOCITY_MULT = {
            0: 1.1,  # Guitar: Strong but not overpowering
            1: 1.0,  # Bass: Moderate boost for punch  
            2: 0.9,  # Drums: REDUCED from 1.5 to 1.1
            3: 0.7   # Other: Slightly lower to not compete
        }
        self.TIMING_JITTER = {
            0: 3,    # Guitar: More humanization for riffs
            1: 2,    # Bass: Moderate timing variation
            2: 1,    # Drums: Tight timing (metal precision)
            3: 2     # Other: Standard variation
        }
        self.SAMPLE_RATE = 44100
        self.MP3_QUALITY = 1  # Higher quality for metal clarity
        
        # Metal-optimized smoothing parameters
        self.VELOCITY_RANDOMIZATION = 20  # More aggressive variation
        self.NOTE_OVERLAP_MS = 5  # Minimal overlap for tight metal sound
        
        # Processing options
        self.CLEANUP_INTERMEDIATE = True
        self.PARALLEL_PROCESSING = False  # Future feature
        
        # Load from config file if provided
        if config_file and config_file.exists():
            self.load_from_file(config_file)
            
        # Ensure output directory exists
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_from_file(self, config_file: Path):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(self, key):
                        if key.endswith('_DIR') or key.endswith('_EXE') or 'PATH' in key:
                            setattr(self, key, Path(value))
                        else:
                            setattr(self, key, value)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def save_to_file(self, config_file: Path):
        """Save current configuration to JSON file"""
        config_data = {}
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                value = getattr(self, attr)
                if isinstance(value, Path):
                    config_data[attr] = str(value)
                else:
                    config_data[attr] = value
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

# -------------------------
# Setup logging
# -------------------------
def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('midi_converter.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# -------------------------
# Enhanced audio rendering
# -------------------------
def midi_to_wav_cli(midi_path: Path, wav_path: Path, soundfont: Path, config: Config) -> bool:
    """Convert MIDI to WAV using FluidSynth with enhanced options and timeout"""
    try:
        # Verify inputs exist
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        if not soundfont.exists():
            raise FileNotFoundError(f"SoundFont not found: {soundfont}")
        if not config.FLUIDSYNTH_EXE.exists():
            raise FileNotFoundError(f"FluidSynth executable not found: {config.FLUIDSYNTH_EXE}")

        logging.info(f"Starting FluidSynth rendering: {midi_path.name} with {soundfont.name}")

        cmd = [
            str(config.FLUIDSYNTH_EXE),
            "-ni",  # No interactive mode
            "-F", str(wav_path),  # Output file
            "-r", str(config.SAMPLE_RATE),  # Sample rate
            "-g", "1.2",  # Slightly higher gain for metal presence
            "-T", "wav",  # Explicitly set output format
            "-z", "512",  # Larger buffer for smoother metal rendering
            "-c", "2",   # Stereo output
            "-R", "0.6", # Moderate reverb (not too muddy for metal)
            "-C", "0.3", # Light chorus (preserve metal clarity)
            str(soundfont),
            str(midi_path),
        ]
        
        # Add timeout to prevent hanging (30 seconds should be enough for most MIDI files)
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        
        if wav_path.exists() and wav_path.stat().st_size > 0:
            logging.info(f"Successfully rendered: {midi_path.name} -> {wav_path.name}")
            return True
        else:
            logging.warning(f"Output file is empty or missing: {wav_path}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"FluidSynth timeout (30s) for {midi_path.name}")
        # Try to clean up partial output
        if wav_path.exists():
            try:
                wav_path.unlink()
            except:
                pass
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"FluidSynth error for {midi_path}: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Error rendering {midi_path}: {e}")
        return False

# -------------------------
# Enhanced audio mixing
# -------------------------
def merge_wavs(wav_files: List[Path], output_wav: Path, config: Config) -> bool:
    """Merge WAV files using ffmpeg with enhanced mixing options and timeout"""
    try:
        # Filter out non-existent or empty files
        valid_files = [f for f in wav_files if f.exists() and f.stat().st_size > 0]
        
        if not valid_files:
            logging.error("No valid WAV files to merge")
            return False
        
        if len(valid_files) == 1:
            # If only one file, just copy it
            import shutil
            shutil.copy2(valid_files[0], output_wav)
            logging.info(f"Single file copied: {valid_files[0].name} -> {output_wav.name}")
            return True

        logging.info(f"Merging {len(valid_files)} WAV files...")

        inputs = []
        for f in valid_files:
            inputs.extend(["-i", str(f)])
        
        # Metal-optimized mixing with tight compression and EQ
        filter_complex = (
            f"amix=inputs={len(valid_files)}:duration=longest:normalize=0[mixed];"
            f"[mixed]acompressor=threshold=0.6:ratio=4:attack=1:release=20[comp];"
            f"[comp]highpass=f=80[hp];"  # Remove muddy low frequencies
            f"[hp]lowpass=f=12000[lp];"  # Remove harsh highs
            f"[lp]alimiter=level_in=1:level_out=0.85[out]"  # Tighter limiting
        )
        
        cmd = [
            "ffmpeg", "-y", "-loglevel", "warning"
        ] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:a", "pcm_s16le",  # Ensure consistent format
            str(output_wav)
        ]
        
        # Add timeout for ffmpeg as well
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        
        if output_wav.exists() and output_wav.stat().st_size > 0:
            logging.info(f"Successfully mixed {len(valid_files)} tracks -> {output_wav.name}")
            return True
        else:
            logging.error(f"Mixed output is empty or missing: {output_wav}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"ffmpeg mixing timeout (60s)")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg mixing error: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Error mixing WAVs: {e}")
        return False

# -------------------------
# Enhanced MP3 conversion
# -------------------------
def wav_to_mp3(wav_path: Path, mp3_path: Path, config: Config) -> bool:
    """Convert WAV to MP3 with enhanced quality settings and timeout"""
    try:
        logging.info(f"Converting to MP3: {wav_path.name} -> {mp3_path.name}")
        
        cmd = [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-i", str(wav_path),
            "-codec:a", "libmp3lame",
            "-qscale:a", str(config.MP3_QUALITY),
            "-joint_stereo", "1",
            str(mp3_path)
        ]
        
        # Add timeout for MP3 conversion
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        
        if mp3_path.exists() and mp3_path.stat().st_size > 0:
            logging.info(f"Successfully converted: {wav_path.name} -> {mp3_path.name}")
            return True
        else:
            logging.error(f"MP3 output is empty or missing: {mp3_path}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"MP3 conversion timeout (30s) for {wav_path.name}")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"MP3 conversion error: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Error converting to MP3: {e}")
        return False

# -------------------------
# Main processing function
# -------------------------
def process_midi_file(midi_file: Path, config: Config, logger) -> bool:
    """
    Process a binary MIDI file for metal genre and save only the final MP3.
    """
    temp_files = []  # Track all temporary files for cleanup
    
    try:
        logger.info(f"Processing (metal): {midi_file.name}")

        def map_metal_channel(msg) -> int:
            # Check for explicit drum channel or drum flag
            if hasattr(msg, 'channel') and msg.channel == 9:  # Standard drum channel
                return 2
            if getattr(msg, "is_drum", False):
                return 2
            prog = getattr(msg, "program", None)
            if prog is not None:
                if 24 <= prog <= 31:  # Guitar family
                    return 0
                if 32 <= prog <= 39:  # Bass family
                    return 1
            # Check if message is on drum channel in original MIDI
            if hasattr(msg, 'channel') and msg.channel == 9:
                return 2
            return 3  # Other accompaniment

        # Load original MIDI
        midi = MidiFile(midi_file)
        logger.info(f"Original MIDI: {len(midi.tracks)} tracks, {midi.ticks_per_beat} ticks/beat")

        # Initialize output MIDI files per channel
        channels_midi: Dict[int, MidiFile] = {
            ch: MidiFile(ticks_per_beat=midi.ticks_per_beat)
            for ch in config.CHANNEL_SOUNDFONTS
        }
        for ch in channels_midi:
            channels_midi[ch].tracks.append(MidiTrack())

        # Process all messages from all tracks
        note_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        for track_idx, track in enumerate(midi.tracks):
            logger.debug(f"Processing track {track_idx} with {len(track)} messages")
            
            # Track current time and channel states per track
            current_time = 0
            current_programs = {}  # channel -> program
            
            for msg in track:
                current_time += msg.time
                
                if msg.is_meta:
                    # Copy meta messages to all channel files
                    for ch in channels_midi:
                        meta_copy = msg.copy()
                        meta_copy.time = msg.time  # Keep original timing
                        channels_midi[ch].tracks[0].append(meta_copy)
                        
                elif hasattr(msg, 'type'):
                    if msg.type == 'program_change':
                        # Track program changes for channel mapping
                        current_programs[msg.channel] = msg.program
                        
                    elif msg.type in ['note_on', 'note_off']:
                        # Determine target channel based on original MIDI channel and program
                        original_channel = getattr(msg, 'channel', 0)
                        
                        # Use program info if available
                        if original_channel in current_programs:
                            temp_msg = type('temp', (), {
                                'program': current_programs[original_channel],
                                'channel': original_channel,
                                'is_drum': original_channel == 9
                            })()
                            target_ch = map_metal_channel(temp_msg)
                        else:
                            # Fallback mapping based on channel
                            if original_channel == 9:
                                target_ch = 2  # Drums
                            elif original_channel < 4:
                                target_ch = original_channel
                            else:
                                target_ch = 3  # Other
                        
                        # Create message for target channel
                        msg_copy = msg.copy()
                        # Drums must stay on channel 9, others go to channel 0
                        msg_copy.channel = 9 if target_ch == 2 else 0
                        msg_copy.time = msg.time  # Preserve original timing
                        
                        # Adjust velocity with humanization
                        if msg_copy.type == "note_on" and getattr(msg_copy, "velocity", 0) > 0:
                            mult = config.VELOCITY_MULT.get(target_ch, 1.0)
                            new_velocity = int(round(msg_copy.velocity * mult))
                            
                            # Add velocity randomization for more natural feel
                            velocity_jitter = random.randint(-config.VELOCITY_RANDOMIZATION, 
                                                           config.VELOCITY_RANDOMIZATION)
                            new_velocity = max(1, min(127, new_velocity + velocity_jitter))
                            msg_copy.velocity = new_velocity
                            note_counts[target_ch] += 1

                        channels_midi[target_ch].tracks[0].append(msg_copy)

        # Log note distribution
        logger.info(f"Note distribution - Guitar: {note_counts[0]}, Bass: {note_counts[1]}, "
                   f"Drums: {note_counts[2]}, Other: {note_counts[3]}")

        # Check if we have any notes at all
        total_notes = sum(note_counts.values())
        if total_notes == 0:
            logger.warning(f"No notes found in {midi_file.name}")
            return False

        # Render each channel that has notes
        channel_wavs: List[Path] = []
        for ch, midi_ch in channels_midi.items():
            if note_counts[ch] == 0:
                logger.info(f"Skipping channel {ch} (no notes)")
                continue
                
            ch_midi_path = config.OUTPUT_DIR / f"{midi_file.stem}_ch{ch}.mid"
            ch_wav_path = config.OUTPUT_DIR / f"{midi_file.stem}_ch{ch}.wav"
            
            # Track temp files for cleanup
            temp_files.extend([ch_midi_path, ch_wav_path])
            
            logger.info(f"Saving channel {ch} MIDI with {note_counts[ch]} notes")
            midi_ch.save(ch_midi_path)

            # Use the specific soundfont for this channel
            sf = config.CHANNEL_SOUNDFONTS[ch]
            if midi_to_wav_cli(ch_midi_path, ch_wav_path, sf, config):
                channel_wavs.append(ch_wav_path)
            else:
                logger.warning(f"Failed to render channel {ch} for {midi_file.name}")

        if not channel_wavs:
            logger.error(f"No channels rendered successfully for {midi_file.name}")
            return False

        # Merge WAVs into final WAV (temporary)
        final_wav = config.OUTPUT_DIR / f"{midi_file.stem}_metal.wav"
        temp_files.append(final_wav)  # Track final WAV for cleanup
        
        if not merge_wavs(channel_wavs, final_wav, config):
            logger.error(f"Failed to merge channels for {midi_file.name}")
            return False

        # Convert final WAV to MP3 (ONLY file we keep)
        final_mp3 = final_wav.with_suffix(".mp3")
        if not wav_to_mp3(final_wav, final_mp3, config):
            logger.error(f"Failed to convert to MP3 for {midi_file.name}")
            return False

        logger.info(f"Successfully completed (metal): {midi_file.name} -> {final_mp3.name}")
        return True

    except Exception as e:
        logger.exception(f"Fatal error processing {midi_file.name}: {e}")
        return False
        
    finally:
        # GUARANTEED cleanup of ALL temporary files
        if config.CLEANUP_INTERMEDIATE:
            for temp_file in temp_files:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                        logger.debug(f"Cleaned up: {temp_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {temp_file}: {e}")

# -------------------------
# Main execution
# -------------------------
def main():
    """Main execution function"""
    # Setup
    logger = setup_logging()
    config = Config()
    
    # Verify dependencies
    dependencies = {
        "FluidSynth": config.FLUIDSYNTH_EXE,
        "MIDI Directory": config.MIDI_DIR,
    }
    
    # Check all soundfonts
    for ch, sf_path in config.CHANNEL_SOUNDFONTS.items():
        if not sf_path.exists():
            logger.error(f"SoundFont for channel {ch} not found: {sf_path}")
            dependencies[f"SoundFont Ch{ch}"] = sf_path
    
    missing_deps = []
    for name, path in dependencies.items():
        if not path.exists():
            logger.error(f"{name} not found at: {path}")
            missing_deps.append(name)
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        return False
    
    # Check for ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        logger.info("ffmpeg found and working")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg not found in PATH")
        return False
    
    # Find MIDI files
    midi_files = list(config.MIDI_DIR.glob("*.mid")) + list(config.MIDI_DIR.glob("*.midi"))
    if not midi_files:
        logger.warning(f"No MIDI files found in {config.MIDI_DIR}")
        return False
    
    logger.info(f"Found {len(midi_files)} MIDI files to process")
    
    # Process files
    successful = 0
    failed = 0
    
    for midi_file in midi_files:
        if process_midi_file(midi_file, config, logger):
            successful += 1
        else:
            failed += 1
    
    # Summary
    logger.info(f"Processing complete: {successful} successful, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)