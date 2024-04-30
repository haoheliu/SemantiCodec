import os
from pydub import AudioSegment

def convert_wav_to_mp3(file_path, output_directory):
    # Load the WAV file
    audio = AudioSegment.from_wav(file_path)
    
    # Create the output file path
    output_file_path = os.path.join(output_directory, os.path.splitext(os.path.basename(file_path))[0] + ".mp3")
    
    # Export the audio segment to MP3 format
    audio.export(output_file_path, format="mp3")
    print(f"Converted and saved: {output_file_path}")

def process_directory(directory):
    # Ensure the output directory structure mirrors the input structure
    for dirpath, dirnames, filenames in os.walk(directory):
        output_directory = dirpath  # Save MP3s in the same directories as the WAVs
        os.makedirs(output_directory, exist_ok=True)
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                file_path = os.path.join(dirpath, filename)
                convert_wav_to_mp3(file_path, output_directory)

# Specify the root directory of your WAV files
root_directory = '/Users/haoheliu/Project/codec-main/audio'
process_directory(root_directory)
