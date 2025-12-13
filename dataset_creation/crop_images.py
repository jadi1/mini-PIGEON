import os
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

def crop_to_square(image_path, output_path, size=224):
    """
    Open an image, crop it to a square, and resize to the specified size.
    
    Args:
        image_path: Path to input image
        output_path: Path to output image
        size: Target square size (default 224)
    """
    try:
        img = Image.open(image_path)
        
        # Get dimensions
        width, height = img.size
        
        # Determine crop size (smallest dimension)
        crop_size = min(width, height)
        
        # Calculate crop coordinates to center the crop
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        # Crop to square
        img_cropped = img.crop((left, top, right, bottom))
        
        # Resize to target size
        img_resized = img_cropped.resize((size, size), Image.Resampling.LANCZOS)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save image
        img_resized.save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def main(input_dir="streetview_images", output_dir="new_streetview_images", size=224):
    """
    Process all images in input directory and save cropped versions to output directory.
    Preserves the directory structure.
    
    Args:
        input_dir: Input directory containing streetview images
        output_dir: Output directory for cropped images
        size: Target square size (default 224)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Output directory: {output_dir}")
    
    success_count = 0
    fail_count = 0
    
    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        # Calculate relative path and create output path
        relative_path = img_file.relative_to(input_path)
        output_img_path = output_path / relative_path
        
        # Process image
        if crop_to_square(str(img_file), str(output_img_path), size):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop street view images to 224x224 squares")
    parser.add_argument("--input", default="data/streetview_images", help="Input directory (default: streetview_images)")
    parser.add_argument("--output", default="data/new_streetview_images", help="Output directory (default: new_streetview_images)")
    parser.add_argument("--size", type=int, default=224, help="Target square size (default: 224)")
    
    args = parser.parse_args()
    main(args.input, args.output, args.size)