import os
from PIL import Image
import numpy as np
import argparse

def convert_jpg_to_raw(input_file, output_file, output_width, output_height):
    """
    Convert a single JPEG file to RAW format.
    """
    try:
        # Open the JPEG image
        img = Image.open(input_file)
        
        # Resize the image
        img_resized = img.resize((output_width, output_height), Image.LANCZOS)
        
        # Convert to RGB format
        img_rgb = img_resized.convert('RGB')
        
        # Get raw pixel data
        raw_data = np.array(img_rgb, dtype=np.uint8)
        
        # Save raw data to file
        with open(output_file, 'wb') as raw_file:
            raw_file.write(raw_data.tobytes())
        
        print(f"Converted: {input_file} -> {output_file}")
    
    except Exception as e:
        print(f"Error converting {input_file}: {e}")

def batch_convert_jpg_to_raw(input_dir, output_dir, output_width, output_height, filename_prefix):
    """
    Batch convert all JPEG files in a directory to RAW format.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, f"{filename_prefix}_{os.path.splitext(filename)[0]}.raw")
            convert_jpg_to_raw(input_file, output_file, output_width, output_height)

    print("Batch conversion completed!")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Batch convert JPEG images to RAW format.")
    parser.add_argument("input_dir", type=str, help="Directory containing input JPEG files")
    parser.add_argument("output_dir", type=str, help="Directory to save converted RAW files")
    parser.add_argument("output_width", type=int, help="Width of the output images")
    parser.add_argument("output_height", type=int, help="Height of the output images")
    parser.add_argument("filename_prefix", type=str, help="Prefix for the output filenames")
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Call the batch conversion function with the arguments
    batch_convert_jpg_to_raw(args.input_dir, args.output_dir, args.output_width, args.output_height, args.filename_prefix)

if __name__ == "__main__":
    main()
