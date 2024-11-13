import os
import torch
import wandb
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

class SegmentationEvaluator:
    def __init__(self, model_path, test_data_path, imgsz=640):
        """
        Initialize evaluator with specific image size
        """
        self.model = YOLO(model_path)
        self.test_data_path = Path(test_data_path)
        self.imgsz = imgsz
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Test data path: {self.test_data_path}")
        print(f"Model input size: {self.imgsz}")

    def process_image(self, image_path):
        """
        Process image to correct size while maintaining aspect ratio
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None
            
        # Get original dimensions
        original_height, original_width = image.shape[:2]
        
        # Calculate scaling to maintain aspect ratio
        scale = self.imgsz / max(original_height, original_width)
        new_height = int(original_height * scale)
        new_width = int(original_width * scale)
        
        # Resize image
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Create square image with padding
        square_image = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        
        # Calculate padding
        y_offset = (self.imgsz - new_height) // 2
        x_offset = (self.imgsz - new_width) // 2
        
        # Place resized image in center
        square_image[y_offset:y_offset + new_height, 
                    x_offset:x_offset + new_width] = resized_image
        
        return square_image, (original_height, original_width, x_offset, y_offset, new_height, new_width)

    def restore_mask_dimensions(self, mask, original_dims):
        """
        Restore mask to original image dimensions
        """
        orig_h, orig_w, x_offset, y_offset, new_h, new_w = original_dims
        
        # Extract the actual mask region (remove padding)
        mask_cropped = mask[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
        
        # Resize to original dimensions
        restored_mask = cv2.resize(mask_cropped.astype(float), (orig_w, orig_h)) > 0.5
        
        return restored_mask

    def save_visualization(self, original_image, masks, conf_scores, save_path):
        """
        Save visualization of segmentation predictions
        """
        # Create a copy of the image for visualization
        vis_image = original_image.copy()
        
        # Create different colors for each instance
        colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)
        
        # Apply each mask with different colors
        for i, (mask, conf) in enumerate(zip(masks, conf_scores)):
            color = colors[i].tolist()
            
            # Create colored overlay for this instance
            overlay = np.zeros_like(original_image)
            overlay[mask] = color
            
            # Blend with main image
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0, vis_image)
            
            # Add confidence score
            y_pos = 30 + (i * 30)
            cv2.putText(
                vis_image,
                f"Instance {i+1}, Conf: {conf:.2f}",
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Save the visualization
        cv2.imwrite(save_path, vis_image)
        return vis_image

    def evaluate(self, output_dir='runs/segment/evaluation'):
        """
        Evaluate model performance on test dataset
        """
        os.makedirs(output_dir, exist_ok=True)
        results_table = wandb.Table(columns=["image", "num_instances", "confidence_scores", "visualization"])
        
        total_instances = 0
        confidence_scores = []
        
        test_images = list(self.test_data_path.glob('pallet_*.jpg'))
        print(f"Found {len(test_images)} pallet images")
        
        for img_path in tqdm(test_images, desc="Evaluating"):
            print(f"\nProcessing image: {img_path}")
            
            # Process image
            square_image, original_dims = self.process_image(img_path)
            if square_image is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            # Keep original image for visualization
            original_image = cv2.imread(str(img_path))
            
            try:
                # Run inference
                results = self.model.predict(
                    source=square_image,
                    task='segment',
                    conf=0.25,
                    iou=0.7,
                    device=self.device
                )[0]
                
                if results.masks is not None:
                    # Get masks and confidence scores
                    masks = results.masks.data.cpu().numpy()
                    conf_scores = results.boxes.conf.cpu().numpy()
                    
                    # Restore masks to original dimensions
                    restored_masks = [
                        self.restore_mask_dimensions(mask, original_dims)
                        for mask in masks
                    ]
                    
                    num_instances = len(restored_masks)
                    total_instances += num_instances
                    confidence_scores.extend(conf_scores)
                    
                    # Create and save visualization
                    vis_path = os.path.join(output_dir, f'vis_{img_path.stem}.png')
                    vis_image = self.save_visualization(
                        original_image, 
                        restored_masks, 
                        conf_scores, 
                        vis_path
                    )
                    
                    # Add to results table
                    results_table.add_data(
                        wandb.Image(original_image),
                        num_instances,
                        np.mean(conf_scores) if len(conf_scores) > 0 else 0,
                        wandb.Image(vis_image)
                    )
                    
                    print(f"Found {num_instances} instances in {img_path.name}")
                    print(f"Confidence scores: {conf_scores}")
                else:
                    print(f"No detections in {img_path.name}")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate summary statistics
        if len(confidence_scores) > 0:
            metrics = {
                "total_images_processed": len(test_images),
                "total_instances_detected": total_instances,
                "avg_instances_per_image": total_instances / len(test_images),
                "mean_confidence": np.mean(confidence_scores),
                "min_confidence": np.min(confidence_scores),
                "max_confidence": np.max(confidence_scores)
            }
            
            # Create confidence score distribution plot
            plt.figure(figsize=(10, 6))
            plt.hist(confidence_scores, bins=20, edgecolor='black')
            plt.title('Distribution of Confidence Scores')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plot_path = os.path.join(output_dir, 'confidence_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            
            # Log to WandB
            wandb.log({
                "results": results_table,
                "metrics": metrics,
                "confidence_distribution": wandb.Image(plot_path)
            })
            
            return metrics
        else:
            print("No valid predictions found")
            return None

def main():
    print("Starting YOLO11 Segmentation Evaluation...")
    
    # Setup paths
    model_path = "runs/segment/train/weights/best.pt"
    test_data_path = "/home/uthira/pallet-detection/data/dataset-segment/test/images"
    
    # Convert to absolute paths
    model_path = os.path.abspath(model_path)
    test_data_path = os.path.abspath(test_data_path)
    
    print(f"Model path: {model_path}")
    print(f"Test data path: {test_data_path}")
    
    # Initialize WandB
    wandb.init(
        project="pallet-segmentation",
        name="yolov11m-seg-evaluation-basic",
        config={
            "model": "yolov11m-seg",
            "task": "segment",
            "evaluation_type": "basic"
        }
    )
    
    try:
        # Initialize evaluator
        evaluator = SegmentationEvaluator(model_path, test_data_path, imgsz=640)
        
        # Run evaluation
        metrics = evaluator.evaluate()
        
        if metrics:
            print("\nEvaluation completed successfully!")
            print("\nResults:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")
        else:
            print("\nEvaluation failed to process any images successfully.")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()