from PIL import Image
import os

def resize_and_convert_to_grayscale(folder, size=(256, 256), output_format="png"):
    """
    Resizes images and converts them to single-channel grayscale,
    regardless of their original number of channels.
    
    Args:
        folder (str): Directory containing the images.
        size (tuple): Target image size (width, height).
        output_format (str): Image format to save ('png', 'jpg', etc.).
    """
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            path = os.path.join(folder, filename)
            try:
                with Image.open(path) as img:
                    # Print original mode for debugging (optional)
                    # print(f"{filename}: {img.mode}")

                    # Convert to grayscale if not already
                    if img.mode != 'L':
                        img = img.convert('L')

                    # Resize image
                    img = img.resize(size, Image.BICUBIC)

                    # Save (overwrite in place)
                    img.save(path, format=output_format.upper())

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
resize_and_convert_to_grayscale("/workspace/project/contrastive-unpaired-translation/datasets/mwir2sim/trainA")
resize_and_convert_to_grayscale("/workspace/project/contrastive-unpaired-translation/datasets/mwir2sim/trainB")


