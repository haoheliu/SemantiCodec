import os
import torchaudio
import soundfile as sf

def normalize_audio_file(file_path, target_peak=1.0):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Compute the peak amplitude
    peak_amplitude = waveform.abs().max()
    
    # Normalize the waveform
    waveform_normalized = waveform / peak_amplitude * target_peak
    
    # Overwrite the original audio file
    sf.write(file_path, waveform_normalized.t().numpy(), sample_rate)

def process_directory(directory):
    # Traverse the directory, and process all files
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(('.wav', '.flac', '.mp3')):
                file_path = os.path.join(dirpath, filename)
                print(f"Normalizing: {file_path}")
                normalize_audio_file(file_path)

# Specify the root directory of your audio files
root_directory = '/Users/haoheliu/Project/codec-main/audio'
process_directory(root_directory)
