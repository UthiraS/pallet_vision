import os
from PIL import Image
import cv2

def print_image_size(image_path):
    """
    Print size and details of a single image
    
    Args:
        image_path (str): Path to the image file
    """
    try:
        # Read image with OpenCV to get dimensions and channels
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1
        
        # Get file size in KB
        file_size = os.path.getsize(image_path) / 1024
        
        print("\nImage Details:")
        print(f"Path: {image_path}")
        print(f"Dimensions: {width}x{height}")
        print(f"Size: {file_size:.2f} KB")
        print(f"Channels: {channels}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

# Example usage
image_path = "/home/uthira/pallet-detection/data/dataset/train/images/pallet_2.jpg"  # Replace with your image path
print_image_size(image_path)