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
    def __init__(self, model_path, test_data_path):
        """
        Initialize evaluator
        Args:
            model_path: Path to trained model weights
            test_data_path: Path to test images directory
        """
        self.model = YOLO(model_path)
        self.test_data_path = Path(test_data_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Test data path: {self.test_data_path}")

    def save_visualization(self, image, masks, conf_scores, save_path):
        """
        Save visualization of segmentation predictions
        """
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Create different colors for each instance
        colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)
        
        # Apply each mask with different colors
        for i, (mask, conf) in enumerate(zip(masks, conf_scores)):
            color = colors[i].tolist()
            
            # Create colored overlay for this instance
            overlay = np.zeros_like(image)
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
                color,
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
        
        # Statistics to track
        total_instances = 0
        confidence_scores = []
        
        # Find all pallet images
        test_images = list(self.test_data_path.glob('pallet_*.jpg'))
        print(f"Found {len(test_images)} pallet images")
        
        for img_path in tqdm(test_images, desc="Evaluating"):
            print(f"\nProcessing image: {img_path}")
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            # Run inference
            try:
                results = self.model.predict(
                    source=str(img_path),
                    task='segment',
                    conf=0.25,
                    iou=0.7,
                    device=self.device
                )[0]
                
                if results.masks is not None:
                    # Get masks and confidence scores
                    masks = results.masks.data.cpu().numpy()
                    conf_scores = results.boxes.conf.cpu().numpy()
                    
                    num_instances = len(masks)
                    total_instances += num_instances
                    confidence_scores.extend(conf_scores)
                    
                    # Convert masks to boolean format
                    binary_masks = [mask > 0 for mask in masks]
                    
                    # Create and save visualization
                    vis_path = os.path.join(output_dir, f'vis_{img_path.stem}.png')
                    vis_image = self.save_visualization(image, binary_masks, conf_scores, vis_path)
                    
                    # Add to results table
                    results_table.add_data(
                        wandb.Image(image),
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
    
    # Verify paths exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data directory not found at {test_data_path}")
    
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
        evaluator = SegmentationEvaluator(model_path, test_data_path)
        
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
        raise e
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()