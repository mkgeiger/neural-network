import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import argparse

def convert_wav_to_raw(input_file, output_file, output_width, output_height):
    """
    Convert a single WAV file to RAW format.
    """
    try:
        # Open the WAV file
        y, sr = librosa.load(input_file)
              
        n_fft = 256
        hop_length = 64
        
        ft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        
        ft_dB = librosa.amplitude_to_db(ft, ref=np.max)
        librosa.display.specshow(ft_dB, sr=sr, hop_length=hop_length);
        #plt.show()  # Show the plot
        
        # Normalize to 0-255 range
        norm_spectrogram = (ft_dB - ft_dB.min()) / (ft_dB.max() - ft_dB.min())
        
        
        # Apply colormap (e.g., 'magma' or 'viridis')
        colormap = plt.colormaps.get_cmap('magma')  # Choose the colormap
        rgb_spectrogram = (colormap(norm_spectrogram)[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB
        
        # Flip image vertically to match `plt.show()`
        rgb_spectrogram = np.flipud(rgb_spectrogram)
        
        # Resize to 18x128 pixels using INTER_CUBIC for best quality
        resized_spectrogram = cv2.resize(rgb_spectrogram, (128, 128), interpolation=cv2.INTER_CUBIC)
    
        # Save as raw 24-bit RGB image
        with open(output_file, "wb") as f:
            f.write(resized_spectrogram.tobytes())
                
        print(f"Converted: {input_file} -> {output_file}")
    
    except Exception as e:
        print(f"Error converting {input_file}: {e}")

def batch_convert_wav_to_raw(input_dir, output_dir, output_width, output_height, filename_prefix):
    """
    Batch convert all WAV files in a directory to RAW format.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.wav')):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, f"{filename_prefix}_{os.path.splitext(filename)[0]}.raw")
            convert_wav_to_raw(input_file, output_file, output_width, output_height)

    print("Batch conversion completed!")
    
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Batch convert JPEG images to RAW format.")
    parser.add_argument("input_dir", type=str, help="Directory containing input WAV files")
    parser.add_argument("output_dir", type=str, help="Directory to save converted RAW files")
    parser.add_argument("output_width", type=int, help="Width of the output images")
    parser.add_argument("output_height", type=int, help="Height of the output images")
    parser.add_argument("filename_prefix", type=str, help="Prefix for the output filenames")
    # Parse command-line arguments
    args = parser.parse_args()
    # Call the batch conversion function with the arguments
    batch_convert_wav_to_raw(args.input_dir, args.output_dir, args.output_width, args.output_height, args.filename_prefix)

if __name__ == "__main__":
    main()

    