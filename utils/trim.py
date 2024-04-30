import os
import torchaudio

def trim_audio_to_match(source_file_path, target_file_path):
    # Load the target audio file (from "groundtruth")
    waveform_target, sample_rate_target = torchaudio.load(target_file_path)
    
    # Determine the duration of the target audio
    target_duration = waveform_target.shape[1] / sample_rate_target
    
    # Load the source audio file
    waveform_source, sample_rate_source = torchaudio.load(source_file_path)
    
    # Calculate the number of samples to match the duration
    num_samples = int(target_duration * sample_rate_source)
    
    # Trim the source waveform to the target duration
    waveform_trimmed = waveform_source[:, :num_samples]
    
    # Overwrite the source file with the trimmed waveform
    torchaudio.save(source_file_path, waveform_trimmed, sample_rate_source)
    print(f"Trimmed and saved: {source_file_path}")

def process_directories(groundtruth_directory, other_directories):
    # Process each file in the groundtruth directory
    for filename in os.listdir(groundtruth_directory):
        groundtruth_file_path = os.path.join(groundtruth_directory, filename)
        # Process each other directory
        for directory in other_directories:
            other_file_path = os.path.join(directory, filename)
            if os.path.exists(other_file_path):
                trim_audio_to_match(other_file_path, groundtruth_file_path)

# Specify the root directory and the groundtruth subdirectory
root_directory = '/Users/haoheliu/Project/codec-main/audio'
groundtruth_directory = os.path.join(root_directory, 'groundtruth')

# List all other directories that need their files trimmed
other_directories = [os.path.join(root_directory, d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d)) and d != 'groundtruth']

process_directories(groundtruth_directory, other_directories)
