
import os
from PIL import Image, ImageEnhance, ImageFilter
import random
import numpy as np
from pathlib import Path

class ImageAugmenter:
    def __init__(self):
        self.target_size = (256, 256)
    
    def add_gaussian_noise(self, image, mean=0, std=25):
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Generate Gaussian noise
        noise = np.random.normal(mean, std, img_array.shape)
        
        # Add noise to image
        noisy_image = img_array + noise
        
        # Clip values to valid range [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_image)
    
    def adjust_brightness(self, image, factor_range=(0.5, 1.5)):
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def adjust_contrast(self, image, factor_range=(0.75, 1.25)):
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def adjust_saturation(self, image, factor_range=(0.6, 1.4)):
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    def apply_gaussian_blur(self, image, radius_range=(0.1, 2.0)):
        radius = random.uniform(*radius_range)
        return image.filter(ImageFilter.GaussianBlur(radius))
    
    def resize_image(self, image):
        return image.resize(self.target_size, Image.Resampling.LANCZOS)
    
    def augment_image(self, image):
        # Apply all augmentations in sequence
        image = self.resize_image(image)
        image = self.add_gaussian_noise(image)
        image = self.adjust_brightness(image)
        image = self.adjust_contrast(image)
        image = self.adjust_saturation(image)
        image = self.apply_gaussian_blur(image)
        return image

def process_directory(input_dir, output_dir, augmenter):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images in the input directory
    for img_path in Path(input_dir).glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            try:
                # Open and augment image
                image = Image.open(img_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Augment image
                augmented_image = augmenter.augment_image(image)
                
                # Save augmented image
                output_path = os.path.join(output_dir, f'aug_{img_path.name}')
                augmented_image.save(output_path, quality=95)
                
                print(f'Processed: {img_path.name}')
            except Exception as e:
                print(f'Error processing {img_path.name}: {str(e)}')

def main():
    # Base directories
    base_dirs = ['/home/uthira/pallet-detection/data/dataset/train', '/home/uthira/pallet-detection/data/dataset/val', '/home/uthira/pallet-detection/data/dataset/test']
    
    # Create augmenter instance
    augmenter = ImageAugmenter()
    
    # Process each base directory
    for base_dir in base_dirs:
        input_dir = os.path.join(base_dir, 'images')
        output_dir = os.path.join(base_dir, 'augmented_images')
        
        if os.path.exists(input_dir):
            print(f'\nProcessing {base_dir} directory...')
            process_directory(input_dir, output_dir, augmenter)
        else:
            print(f'\nDirectory not found: {input_dir}')

if __name__ == "__main__":
    main()