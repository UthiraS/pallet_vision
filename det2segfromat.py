

# """

# !git clone https://github.com/facebookresearch/segment-anything-2.git
# # %cd {HOME}/segment-anything-2
# !pip install -e . -q

# !pip install -q supervision jupyter_bbox_widget

# """### Download SAM2 checkpoints

# *** SAM2 is available in 4 different model sizes ranging from the lightweight "sam2_hiera_tiny" (38.9M parameters) to the more powerful "sam2_hiera_large" (224.4M parameters).
# """

# !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt -P /content/weights/checkpoints
# !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt -P /content/weights/checkpoints
# !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt -P /content/weights/checkpoints
# !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P /content/weights/checkpoints

# """

import base64
import cv2
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from tqdm import tqdm
from skimage import measure
import numpy as np
import supervision as sv

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2_pallets import *

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = f"/content/weights/checkpoints/sam2_hiera_large.pt"
CONFIG = "sam2_hiera_l.yaml"

sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)

predictor = SAM2ImagePredictor(sam2_model)





def convert_mask_to_yolo_polygon(mask, image_shape):
    """Convert binary mask to YOLO polygon format"""
    # Find contours
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
    if not contours:
        return None

    # Get largest contour
    contour = max(contours, key=len)

    # Reduce points (maintain roughly 20-30 points)
    n_points = len(contour)
    step = max(1, n_points // 20)
    contour = contour[::step]

    # Convert to normalized coordinates
    h, w = image_shape[:2]
    polygon = []
    for point in contour:
        y, x = point
        x_norm = x / w
        y_norm = y / h
        polygon.extend([x_norm, y_norm])

    # Add first point again to close the polygon
    polygon.extend([polygon[0], polygon[1]])

    return polygon

def save_yolo_segments(masks, output_dir, image_name, image_shape):
    """Save segmentation masks in YOLO format"""
    segment_dir = output_dir / 'segment_labels'
    segment_dir.mkdir(exist_ok=True)

    output_path = segment_dir / f"{image_name}.txt"

    with open(output_path, 'w') as f:
        for mask in masks:
            polygon = convert_mask_to_yolo_polygon(mask, image_shape)
            if polygon:
                # Format: class_id x1 y1 x2 y2 ...
                line = "0 " + " ".join([f"{coord:.16f}" for coord in polygon])
                f.write(line + '\n')

def process_image(predictor, image_rgb, image_bgr, boxes):
    """Process single image with SAM2 and create visualization."""
    try:
        # Set image for prediction
        predictor.set_image(image_rgb)

        # Get masks
        masks, scores, logits = predictor.predict(
            box=boxes,
            multimask_output=False
        )

        # Handle mask dimensions properly
        if len(boxes) == 1:
            masks = np.array([masks[0]])  # Ensure 3D
        else:
            masks = np.squeeze(masks)
            if len(masks.shape) == 2:
                masks = np.array([masks])  # Make 3D

        # Convert to boolean for visualization
        masks_bool = masks.astype(bool)

        # Create annotators
        box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

        # Create detections
        detections = sv.Detections(
            xyxy=boxes,
            mask=masks_bool
        )

        # Create visualizations
        source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

        # Combine images
        combined_image = np.hstack([source_image, segmented_image])

        return combined_image, masks, scores

    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        raise

def save_masks(masks, scores, output_dir, image_name, image_shape):
    """Save masks, scores, and YOLO segments"""
    try:
        # Create output directories
        masks_dir = output_dir / 'masks'
        masks_dir.mkdir(exist_ok=True)

        # Ensure masks is 3D
        if len(masks.shape) == 2:
            masks = np.array([masks])

        # Save individual masks
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_path = masks_dir / f"{image_name}_mask_{i}.npy"
            np.save(str(mask_path), mask.astype(np.uint8))

        # Save scores
        scores_path = masks_dir / f"{image_name}_scores.npy"
        np.save(str(scores_path), scores)

        # Save combined mask
        combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
        for i, mask in enumerate(masks, 1):
            combined_mask[mask.astype(bool)] = i
        combined_path = masks_dir / f"{image_name}_combined.npy"
        np.save(str(combined_path), combined_mask)

        # Save YOLO format segmentation
        save_yolo_segments(masks, output_dir, image_name, image_shape)

    except Exception as e:
        print(f"Error in save_masks: {str(e)}")
        raise

def main(input_dir, labels_dir, output_dir, checkpoint_path):
    """Main function to process all images in a directory."""
    # Setup directories
    input_dir = Path(input_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create required directories
    (output_dir / 'masks').mkdir(exist_ok=True)
    (output_dir / 'segment_labels').mkdir(exist_ok=True)

    # Setup model
    model, device = setup_sam2_model(checkpoint_path)
    predictor = SAM2ImagePredictor(model)

    # Enable mixed precision
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Process images
    label_files = sorted(labels_dir.glob("pallet_*.txt"))

    for label_path in tqdm(label_files, desc="Processing images"):
        try:
            # Get corresponding image path
            image_filename = label_path.stem + ".jpg"
            image_path = input_dir / image_filename

            if not image_path.exists():
                print(f"No image found for {label_path.name}, skipping...")
                continue

            # Load image and boxes
            image_bgr, image_rgb = load_image(image_path)
            boxes = load_yolo_annotations(label_path)

            # Skip if no valid boxes
            if len(boxes) == 0:
                print(f"No valid boxes found in {label_path.name}, skipping...")
                continue

            # Process image
            result_image, masks, scores = process_image(predictor, image_rgb, image_bgr, boxes)

            # Save results
            output_path = output_dir / f"{label_path.stem}_segmented.jpg"
            cv2.imwrite(str(output_path), result_image)

            # Save masks, scores, and YOLO segments
            save_masks(masks, scores, output_dir, label_path.stem, image_bgr.shape)

        except Exception as e:
            print(f"Error processing {label_path.name}: {str(e)}")
            continue

if __name__ == "__main__":
    HOME = os.getcwd()
    CHECKPOINT = f"/content/weights/checkpoints/sam2_hiera_large.pt"
    INPUT_DIR = "/content/drive/MyDrive/data/Pallets"
    LABELS_DIR = "/content/drive/MyDrive/data/Labels_Pallets"
    OUTPUT_DIR = "/content/drive/MyDrive/data/sam2_Pallets"

    main(INPUT_DIR, LABELS_DIR, OUTPUT_DIR, CHECKPOINT)
