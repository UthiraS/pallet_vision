import os
import torch
from ultralytics import YOLO
import cv2
import supervision as sv
import argparse
import numpy as np 
def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='YOLO Detection Single Image Inference')
    parser.add_argument('--weights', type=str, required=True,
                      help='Path to model weights file (e.g., best.pt)')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to test image')
    return parser.parse_args()

class SingleImageDetection:
    def __init__(self, model_path):
        """Initialize as before"""
        self.model = YOLO(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Detection using device: {self.device}")
        
    def display_results(self, image, results):
        """Display detection results"""
        h, w = image.shape[:2]
        grid = np.zeros((h + 30, w*2, 3), dtype=np.uint8)
        
        # Original image (left)
        grid[0:h, :w] = image
        
        # Detections (right)
        if results.boxes is not None and len(results.boxes):
            # Draw boxes on copy of image
            annotated = image.copy()
            for box in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = box
                cv2.rectangle(annotated, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                cv2.putText(annotated, 
                           f'conf: {conf:.2f}', 
                           (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
            grid[0:h, w:] = annotated
        else:
            grid[0:h, w:] = image
            
        cv2.imshow('Detection Results', grid)
        
    def run_inference(self, image):
        """Run detection inference"""
        try:
            results = self.model.predict(
                source=image,
                conf=0.25,
                iou=0.7,
                max_det=300,
                device=self.device
            )[0]
            
            self.display_results(image, results)
            
            if results.boxes is not None:
                print(f"Detection found {len(results.boxes)} instances")
            else:
                print("No detection instances found")
                
        except Exception as e:
            print(f"Error in detection: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    print("Starting YOLO Detection Single Image Inference...")
    
    # Parse command line arguments
    args = parse_args()
    
    # Convert to absolute paths
    model_path = os.path.abspath(args.weights)
    image_path = os.path.abspath(args.image)
    
    print(f"\nUsing paths:")
    print(f"Model weights: {model_path}")
    print(f"Test image: {image_path}")
    
    try:
        # Initialize detector
        detector = SingleImageDetection(model_path)
        
        # Run inference
        detector.run_inference(image_path)
        
        print("\nInference completed!")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()