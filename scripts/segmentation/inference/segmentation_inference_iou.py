import os
import torch
import wandb
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='YOLO Segmentation Evaluation')
    
    # Required arguments
    parser.add_argument('--weights', type=str, required=True,
                      help='Path to model weights file (e.g., best.pt)')
    parser.add_argument('--test-images', type=str, required=True,
                      help='Path to test images directory')
    parser.add_argument('--masks', type=str, required=True,
                      help='Path to ground truth masks directory')
    
    return parser.parse_args()

class SegmentationEvaluator:
    def __init__(self, model_path, test_data_path, mask_data_path, imgsz=640):
        """
        Initialize evaluator
        Args:
            model_path: Path to model weights
            test_data_path: Path to test images
            mask_data_path: Path to ground truth masks
            imgsz: Model input size
        """
        self.model = YOLO(model_path)
        self.test_data_path = Path(test_data_path)
        self.mask_data_path = Path(mask_data_path)
        self.imgsz = imgsz
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Test data path: {self.test_data_path}")
        print(f"Mask data path: {self.mask_data_path}")

    def load_ground_truth_mask(self, img_path):
        """
        Load corresponding ground truth mask
        """
        try:
            mask_path = self.mask_data_path / f"{img_path.stem}_combined.npy"
            if mask_path.exists():
                mask = np.load(str(mask_path))
                binary_mask = mask > 0
                print(f"Loaded GT mask shape: {binary_mask.shape}")
                return binary_mask
            else:
                print(f"No GT mask found at {mask_path}")
                return None
        except Exception as e:
            print(f"Error loading GT mask: {str(e)}")
            return None

    def calculate_iou(self, pred_mask, gt_mask):
        """
        Calculate IoU between predicted and ground truth masks
        """
        if pred_mask is None or gt_mask is None:
            return 0
        
        intersection = np.logical_and(pred_mask, gt_mask)
        union = np.logical_or(pred_mask, gt_mask)
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        return iou

    def calculate_mask_metrics(self, pred_masks, gt_mask):
        """
        Calculate IoU and other metrics for all predicted masks
        """
        if gt_mask is None or not pred_masks:
            return None
            
        # Calculate IoU for each predicted mask
        ious = []
        for pred_mask in pred_masks:
            iou = self.calculate_iou(pred_mask, gt_mask)
            ious.append(iou)
            
        metrics = {
            'max_iou': max(ious) if ious else 0,
            'mean_iou': np.mean(ious) if ious else 0,
            'all_ious': ious
        }
        return metrics

    def save_visualization(self, original_image, pred_masks, gt_mask, conf_scores, metrics, save_path):
        """
        Save visualization with predictions, ground truth, and metrics
        """
        # Create a 2x2 grid visualization
        h, w = original_image.shape[:2]
        grid = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        
        # Original image (top-left)
        grid[:h, :w] = original_image
        
        # Ground truth overlay (top-right)
        gt_vis = original_image.copy()
        if gt_mask is not None:
            gt_overlay = np.zeros_like(original_image)
            gt_overlay[gt_mask] = [0, 255, 0]  # Green for ground truth
            gt_vis = cv2.addWeighted(gt_vis, 0.7, gt_overlay, 0.3, 0)
        grid[:h, w:] = gt_vis
        
        # Predictions overlay (bottom-left)
        pred_vis = original_image.copy()
        colors = np.random.randint(0, 255, size=(len(pred_masks), 3), dtype=np.uint8)
        for i, (mask, conf) in enumerate(zip(pred_masks, conf_scores)):
            color = colors[i].tolist()
            overlay = np.zeros_like(original_image)
            overlay[mask] = color
            pred_vis = cv2.addWeighted(pred_vis, 0.7, overlay, 0.3, 0)
            
            # Add confidence and IoU scores
            y_pos = 30 + (i * 30)
            text = f"Inst {i+1}, Conf: {conf:.2f}"
            if metrics and metrics['all_ious']:
                text += f", IoU: {metrics['all_ious'][i]:.2f}"
            cv2.putText(pred_vis, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        grid[h:, :w] = pred_vis
        
        # Difference visualization (bottom-right)
        diff_vis = original_image.copy()
        if gt_mask is not None:
            for pred_mask in pred_masks:
                diff_overlay = np.zeros_like(original_image)
                # True Positive (Green)
                diff_overlay[np.logical_and(pred_mask, gt_mask)] = [0, 255, 0]
                # False Positive (Red)
                diff_overlay[np.logical_and(pred_mask, ~gt_mask)] = [0, 0, 255]
                # False Negative (Blue)
                diff_overlay[np.logical_and(~pred_mask, gt_mask)] = [255, 0, 0]
                diff_vis = cv2.addWeighted(diff_vis, 0.7, diff_overlay, 0.3, 0)
        grid[h:, w:] = diff_vis
        
        # Add metric text
        if metrics:
            cv2.putText(grid, f"Max IoU: {metrics['max_iou']:.3f}", (w+10, h+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(grid, f"Mean IoU: {metrics['mean_iou']:.3f}", (w+10, h+70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(save_path, grid)
        return grid

    def evaluate(self, output_dir='runs/segment/evaluation'):
        os.makedirs(output_dir, exist_ok=True)
        results_table = wandb.Table(columns=["image", "num_instances", "max_iou", "mean_iou", "visualization"])
        
        total_instances = 0
        all_ious = []
        confidence_scores = []
        
        test_images = list(self.test_data_path.glob('pallet_*.jpg'))
        print(f"Found {len(test_images)} pallet images")
        
        for img_path in tqdm(test_images, desc="Evaluating"):
            print(f"\nProcessing image: {img_path}")
            
            # Load image and ground truth
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Failed to load image: {img_path}")
                continue
                
            gt_mask = self.load_ground_truth_mask(img_path)
            
            try:
                # Run inference
                results = self.model.predict(
                    source=image,
                    task='segment',
                    conf=0.25,
                    iou=0.7,
                    device=self.device
                )[0]
                
                if results.masks is not None:
                    # Get masks and confidence scores
                    pred_masks = results.masks.data.cpu().numpy()
                    conf_scores = results.boxes.conf.cpu().numpy()
                    
                    # Resize masks to match image dimensions
                    restored_masks = []
                    for mask in pred_masks:
                        if mask.shape[:2] != image.shape[:2]:
                            mask = cv2.resize(mask.astype(float), 
                                           (image.shape[1], image.shape[0])) > 0.5
                        restored_masks.append(mask)
                    
                    # Calculate metrics
                    metrics = self.calculate_mask_metrics(restored_masks, gt_mask)
                    if metrics:
                        all_ious.extend(metrics['all_ious'])
                    
                    num_instances = len(restored_masks)
                    total_instances += num_instances
                    confidence_scores.extend(conf_scores)
                    
                    # Create visualization
                    vis_path = os.path.join(output_dir, f'vis_{img_path.stem}.png')
                    vis_image = self.save_visualization(
                        image, restored_masks, gt_mask, 
                        conf_scores, metrics, vis_path
                    )
                    
                    # Log results
                    results_table.add_data(
                        wandb.Image(image),
                        num_instances,
                        metrics['max_iou'] if metrics else 0,
                        metrics['mean_iou'] if metrics else 0,
                        wandb.Image(vis_image)
                    )
                    
                    print(f"Found {num_instances} instances in {img_path.name}")
                    if metrics:
                        print(f"Max IoU: {metrics['max_iou']:.4f}")
                        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
                else:
                    print(f"No detections in {img_path.name}")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate summary statistics
        if len(all_ious) > 0:
            metrics = {
                "total_images_processed": len(test_images),
                "total_instances_detected": total_instances,
                "avg_instances_per_image": total_instances / len(test_images),
                "mean_iou_overall": np.mean(all_ious),
                "max_iou_overall": np.max(all_ious),
                "min_iou_overall": np.min(all_ious),
                "median_iou": np.median(all_ious),
                "mean_confidence": np.mean(confidence_scores),
            }
            
            # Create IoU distribution plot
            plt.figure(figsize=(10, 6))
            plt.hist(all_ious, bins=20, edgecolor='black')
            plt.title('Distribution of IoU Scores')
            plt.xlabel('IoU Score')
            plt.ylabel('Frequency')
            plot_path = os.path.join(output_dir, 'iou_distribution.png')
            plt.savefig(plot_path)
            plt.close()
            
            # Log to WandB
            wandb.log({
                "results": results_table,
                "metrics": metrics,
                "iou_distribution": wandb.Image(plot_path)
            })
            
            return metrics
        else:
            print("No valid predictions found")
            return None

def main():
    print("Starting YOLO Segmentation Evaluation...")
    
    # Parse command line arguments
    args = parse_args()
    
    # Convert to absolute paths
    model_path = os.path.abspath(args.weights)
    test_data_path = os.path.abspath(args.test_images)
    mask_data_path = os.path.abspath(args.masks)
    
    print(f"\nUsing paths:")
    print(f"Model weights: {model_path}")
    print(f"Test images: {test_data_path}")
    print(f"Mask data: {mask_data_path}")
    
    # Initialize WandB
    wandb.init(
        project="pallet-segmentation",
        name="yolov11m-seg-evaluation-iou",
        config={
            "model": "yolov11m-seg",
            "task": "segment",
            "evaluation_type": "mask_iou"
        }
    )
    
    try:
        # Initialize evaluator
        evaluator = SegmentationEvaluator(model_path, test_data_path, mask_data_path)
        
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