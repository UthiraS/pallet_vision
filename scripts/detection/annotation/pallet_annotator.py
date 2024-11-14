import os
from PIL import Image, ImageEnhance, ImageFilter
import random
import numpy as np
from pathlib import Path
import cv2
from roboflow import Roboflow
import supervision as sv

class ImageAugmenter:
    def __init__(self):
        self.target_size = (640, 640)
    
    def add_gaussian_noise(self, image, mean=0, std=25):
        img_array = np.array(image)
        noise = np.random.normal(mean, std, img_array.shape)
        noisy_image = img_array + noise
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
        image = self.resize_image(image)
        image = self.add_gaussian_noise(image)
        image = self.adjust_brightness(image)
        image = self.adjust_contrast(image)
        image = self.adjust_saturation(image)
        image = self.apply_gaussian_blur(image)
        return image

class_names = {
    1: "pallet"
}

def process_with_roboflow(input_dir: str, output_dir: str, labels_dir: str):
    """
    Process augmented images using Roboflow for object detection
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Initialize Roboflow model
    rf = Roboflow(api_key="XQoeI48zI9DJYy5JrGwj")
    model = rf.workspace().project("pallets-people").version(1).model

    # Create supervision annotator
    bounding_box_annotator = sv.BoundingBoxAnnotator()

    # Get all images
    image_files = [f for f in os.listdir(input_dir)
                  if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

    with sv.ImageSink(target_dir_path=output_dir, overwrite=True) as sink:
        for image_file in image_files:
            try:
                image_path = os.path.join(input_dir, image_file)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Failed to load image: {image_file}")
                    continue

                print(f"Processing {image_file}...")

                predictions = model.predict(
                    image_path,
                    confidence=40,
                    overlap=30
                ).json()
                
                boxes = []
                confidences = []
                class_ids = []

                for pred in predictions['predictions']:
                    try:
                        x = pred['x'] * image.shape[1]
                        y = pred['y'] * image.shape[0]
                        width = pred['width'] * image.shape[1]
                        height = pred['height'] * image.shape[0]

                        x1 = int(x - width/2)
                        y1 = int(y - height/2)
                        x2 = int(x + width/2)
                        y2 = int(y + height/2)

                        boxes.append([x1, y1, x2, y2])
                        confidences.append(pred['confidence'])
                        class_ids.append(pred.get('class_id', 0))

                    except KeyError as e:
                        print(f"Missing key in prediction: {e}")
                        continue

                # Save YOLO format labels
                base_name = Path(image_file).stem
                label_file = os.path.join(labels_dir, f"{base_name}.txt")

                height, width = image.shape[:2]
                with open(label_file, 'w') as f:
                    for box, class_id in zip(boxes, class_ids):
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / (2 * width)
                        center_y = (y1 + y2) / (2 * height)
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n")

                # Create annotated image
                detections = sv.Detections.from_inference(predictions)
                labels = [
                    f"{class_names[class_id]} {confidence:0.2f}"
                    for class_id, confidence in zip(class_ids, confidences)
                ]

                annotated_frame = bounding_box_annotator.annotate(
                    scene=image.copy(),
                    detections=detections
                )

                sink.save_image(
                    image=annotated_frame,
                    image_name=f"detected_{image_file}"
                )

                print(f"Processed {image_file}: Found {len(boxes)} objects")

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue

def main():
    # Define base directories
    base_dirs = ['/home/uthira/pallet-detection/data/dataset/train', '/home/uthira/pallet-detection/data/dataset/val', '/home/uthira/pallet-detection/data/dataset/test']
    
    # Create augmenter instance
    augmenter = ImageAugmenter()
    
    # Process each base directory
    for base_dir in base_dirs:
        # Directories for augmentation
        input_dir = os.path.join(base_dir, 'images')
        augmented_dir = os.path.join(base_dir, 'augmented_images')
        
        # Directories for Roboflow processing
        output_dir = os.path.join(base_dir, 'annotated_images')
        labels_dir = os.path.join(base_dir, 'labels')
        
        if os.path.exists(input_dir):
            print(f'\nProcessing {base_dir} directory...')
            
            # Step 1: Augment images
            os.makedirs(augmented_dir, exist_ok=True)
            for img_path in Path(input_dir).glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    try:
                        image = Image.open(img_path)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        augmented_image = augmenter.augment_image(image)
                        output_path = os.path.join(augmented_dir, f'aug_{img_path.name}')
                        augmented_image.save(output_path, quality=95)
                        print(f'Augmented: {img_path.name}')
                    except Exception as e:
                        print(f'Error augmenting {img_path.name}: {str(e)}')
            
            # Step 2: Process augmented images with Roboflow
            print(f'\nRunning Roboflow detection on augmented images in {base_dir}...')
            process_with_roboflow(augmented_dir, output_dir, labels_dir)
        else:
            print(f'\nDirectory not found: {input_dir}')

if __name__ == "__main__":
    main()