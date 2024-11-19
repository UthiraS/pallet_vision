import os
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import argparse

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='YOLO Segmentation Single Image Inference')
    parser.add_argument('--weights', type=str, required=True,
                      help='Path to model weights file (e.g., best.pt)')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to test image')
    return parser.parse_args()

class SingleImageSegInference:
    def __init__(self, model_path):
        """Initialize as before"""
        self.model = YOLO(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Segmentation using device: {self.device}")

    def display_results(self, original_image, pred_masks=None, conf_scores=None):
        """Display segmentation results"""
        h, w = original_image.shape[:2]
        grid = np.zeros((h + 30, w*2, 3), dtype=np.uint8)
        
        # Original image (left)
        grid[0:h, :w] = original_image
        
        # Right side - segmentation
        if pred_masks is not None and len(pred_masks) > 0:
            pred_vis = original_image.copy()
            colors = np.random.randint(0, 255, size=(len(pred_masks), 3), dtype=np.uint8)
            
            combined_overlay = np.zeros_like(original_image)
            for i, mask in enumerate(pred_masks):
                color = colors[i].tolist()
                combined_overlay[mask] = color
            
            pred_vis = cv2.addWeighted(pred_vis, 0.7, combined_overlay, 0.3, 0)
            grid[0:h, w:] = pred_vis
        else:
            grid[0:h, w:] = original_image
        
        cv2.imshow('Segmentation Results', grid)

    def run_inference(self, image):
        """Run segmentation inference"""
        try:
            results = self.model.predict(
                source=image,
                task='segment',
                conf=0.25,
                iou=0.7,
                device=self.device
            )[0]
            
            if results.masks is not None:
                pred_masks = results.masks.data.cpu().numpy()
                conf_scores = results.boxes.conf.cpu().numpy()
                
                restored_masks = []
                for mask in pred_masks:
                    if mask.shape[:2] != image.shape[:2]:
                        mask = cv2.resize(mask.astype(float), 
                                        (image.shape[1], image.shape[0])) > 0.5
                    restored_masks.append(mask)
                
                self.display_results(image, restored_masks, conf_scores)
                print(f"Segmentation found {len(restored_masks)} instances")
            else:
                self.display_results(image)
                print("No segmentation instances found")
                
        except Exception as e:
            print(f"Error in segmentation: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    print("Starting YOLO Segmentation Single Image Inference...")
    
    # Parse command line arguments
    args = parse_args()
    
    # Convert to absolute paths
    model_path = os.path.abspath(args.weights)
    image_path = os.path.abspath(args.image)
    
    print(f"\nUsing paths:")
    print(f"Model weights: {model_path}")
    print(f"Test image: {image_path}")
    
    try:
        # Initialize inferencer
        inferencer = SingleImageSegInference(model_path)
        
        # Run inference
        inferencer.run_inference(image_path)
        
        print("\nInference completed!")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()