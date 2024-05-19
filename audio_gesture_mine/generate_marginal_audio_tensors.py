import os
import numpy as np
import wave

def process_wav_file(wav_path, chunk_size=9393300, buffer_size=2048):
    """Process a .wav file into chunks and convert to sequence of NumPy arrays with buffer size 2048."""
    with wave.open(wav_path, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()

        # Read the audio data
        audio_data = wav_file.readframes(n_frames)
        audio_samples = np.frombuffer(audio_data, dtype=np.int16)

        # Ensure that we are dealing with mono audio
        if n_channels > 1:
            audio_samples = audio_samples.reshape(-1, n_channels)
            audio_samples = audio_samples.mean(axis=1).astype(np.int16)

        # Split audio data into chunks
        total_samples = audio_samples.shape[0]
        chunks = [audio_samples[i:i+chunk_size] for i in range(0, total_samples, chunk_size)]

        # Process each chunk into buffers of size 2048
        chunk_arrays = []
        for chunk in chunks:
            num_buffers = len(chunk) // buffer_size
            truncated_samples = chunk[:num_buffers * buffer_size]
            audio_buffers = truncated_samples.reshape(num_buffers, buffer_size)
            chunk_arrays.append(audio_buffers)

        return chunk_arrays

def process_directory(input_dir, output_dir, chunk_size=9393300, buffer_size=2048):
    """Process all .wav files in the input directory and save the resulting arrays to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            wav_path = os.path.join(input_dir, filename)
            chunk_arrays = process_wav_file(wav_path, chunk_size=chunk_size, buffer_size=buffer_size)

            # Save each chunk array to the output directory
            base_filename = filename.replace(".wav", "")
            for i, audio_array in enumerate(chunk_arrays):
                output_filename = f"{base_filename}_chunk_{i+1}_buffers.npy"
                output_path = os.path.join(output_dir, output_filename)
                np.save(output_path, audio_array)
                print(f"Processed {filename} chunk {i+1} and saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process .wav files into sequence torch tensors of buffers.")
    parser.add_argument("--input_dir", default = '/Users/nikny/local_data/marginal_audio', type=str, help="Path to the input directory containing .wav files.")
    parser.add_argument("--output_dir", default = '/Users/nikny/local_data/marginal_audio_tensors', type=str, help="Path to the output directory where tensor files will be saved.")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)
