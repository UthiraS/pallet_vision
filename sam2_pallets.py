import os
import cv2
import torch
import numpy as np
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def load_sam2_model(checkpoint_path, config="sam2_hiera_l.yaml", device="cuda"):
    """Initialize SAM2 model"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    sam2_model = build_sam2(config, checkpoint_path, device=device, apply_postprocessing=False)
    return sam2_model, device

def yolo_to_xyxy(center_x, center_y, width, height, img_width, img_height):
    """Convert YOLO format (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max)"""
    # Convert from normalized coordinates to absolute coordinates
    center_x *= img_width
    center_y *= img_height
    width *= img_width
    height *= img_height
    
    x_min = center_x - (width / 2)
    y_min = center_y - (height / 2)
    x_max = center_x + (width / 2)
    y_max = center_y + (height / 2)
    
    # Ensure coordinates are within image bounds
    x_min = max(0, min(img_width, x_min))
    y_min = max(0, min(img_height, y_min))
    x_max = max(0, min(img_width, x_max))
    y_max = max(0, min(img_height, y_max))
    
    return [x_min, y_min, x_max, y_max]

def load_annotations(annotation_path, img_width, img_height):
    """Load YOLO format annotations and convert to xyxy format"""
    boxes = []
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                try:
                    # Parse YOLO format: class_id center_x center_y width height
                    values = line.strip().split()
                    if len(values) != 5:
                        continue
                        
                    # Extract coordinates (ignore class_id)
                    _, center_x, center_y, width, height = map(float, values)
                    
                    # Convert to absolute coordinates
                    box = yolo_to_xyxy(
                        center_x, center_y,
                        width, height,
                        img_width, img_height
                    )
                    
                    boxes.append(box)
                    
                except ValueError as e:
                    print(f"Error parsing line: {line.strip()}")
                    continue
                
        if not boxes:
            print(f"Warning - No valid boxes found in {annotation_path}")
            return None
            
        return np.array(boxes)
        
    except Exception as e:
        print(f"Error reading annotation file {annotation_path}: {str(e)}")
        return None

def process_single_image(predictor, image_path, annotation_path):
    """Process a single image with its annotations"""
    # Read image first to get dimensions
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Get image dimensions
    img_height, img_width = image_bgr.shape[:2]
    
    # Load annotations with image dimensions
    boxes = load_annotations(annotation_path, img_width, img_height)
    if boxes is None or len(boxes) == 0:
        raise ValueError("No valid boxes found in annotation file")
    
    # Convert to RGB for SAM2
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Set image in predictor
    predictor.set_image(image_rgb)
    
    # Get predictions
    masks, scores, logits = predictor.predict(
        box=boxes,
        multimask_output=False
    )
    
    # Handle mask dimensions
    if len(boxes) == 1:
        masks = np.expand_dims(masks, axis=0)
    else:
        masks = np.squeeze(masks)
    
    return masks, scores, image_bgr

def save_results(output_dir, image_name, masks, original_image):
    """Save segmentation results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual masks as binary images
    for idx, mask in enumerate(masks):
        mask_path = os.path.join(output_dir, f"{image_name}_mask_{idx}.png")
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
    
    # Visualize masks on original image
    visualization = original_image.copy()
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
    
    # Create semi-transparent overlay
    overlay = visualization.copy()
    for idx, mask in enumerate(masks):
        color = colors[idx % len(colors)]
        mask_area = mask.astype(bool)
        overlay[mask_area] = color
    
    # Blend the overlay with the original image
    alpha = 0.5
    visualization = cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0)
    
    # Save visualization
    vis_path = os.path.join(output_dir, f"{image_name}_visualization.jpg")
    cv2.imwrite(vis_path, visualization)
def main():
    # Configuration
    checkpoint_path = "/home/uthira/pallet-detection/segment-anything-2/weights/checkpoints/sam2_hiera_large.pt"  # Update with your checkpoint path
    pallet_dir = "data/Pallets"
    annotation_dir = "data/Labels_Pallets"
    output_dir = "data/sam2_Pallets"
    
    # Initialize model
    sam2_model, device = load_sam2_model(checkpoint_path)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Process each image
    for image_path in Path(pallet_dir).glob("*.jpg"):  # Adjust file extension if needed
        image_name = image_path.stem
        annotation_path = Path(annotation_dir) / f"{image_name}.txt"  # Adjust extension if needed
        
        if not annotation_path.exists():
            print(f"No annotation found for {image_name}, skipping...")
            continue
        
        print(f"Processing {image_name}...")
        
        try:
            # Load annotations with debug info
            boxes = load_annotations(annotation_path)
            
            # Process image
            masks, scores, original_image = process_image(str(image_path), boxes, predictor)
            
            # Save results
            save_results(output_dir, image_name, masks, original_image)
            
            print(f"Successfully processed {image_name}")
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()