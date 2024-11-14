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
        """
        Initialize inference
        Args:
            model_path: Path to model weights
        """
        self.model = YOLO(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize supervision annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def display_results(self, image, results):
        """
        Display inference results using cv2.imshow without waiting for key press
        """
        # Create visualization with original and predictions side by side
        h, w = image.shape[:2]
        grid = np.zeros((h + 30, w*2, 3), dtype=np.uint8)  # Added space for captions
        
        # Original image (left)
        grid[0:h, :w] = image
        
        # Add caption for original image
        cv2.putText(grid, 'Original Image', (10, h + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detections (right)
        if results.boxes is not None:
            # Convert results to supervision format
            detections = sv.Detections.from_ultralytics(results)
            
            # Annotate image with boxes and labels
            annotated_image = self.box_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections)
            
            grid[0:h, w:] = annotated_image
        else:
            # If no detections, show original image on right side
            grid[0:h, w:] = image
        
        # Add caption for detection
        cv2.putText(grid, 'Detection Output', (w + 10, h + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display without waiting
        cv2.imshow('Detection Results', grid)
        cv2.waitKey(1)  # Changed from waitKey(0) to waitKey(1)

    def run_inference(self, image):
        """
        Run inference on single image
        """
        try:
            # Run inference
            results = self.model.predict(
                source=image,
                conf=0.25,
                iou=0.7,
                max_det=300,
                device=self.device,
                half=True
            )[0]
            
            # Always display results, whether detections found or not
            self.display_results(image, results)
            
            if results.boxes is not None and len(results.boxes) > 0:
                # Print results (optional, you might want to remove this for continuous operation)
                print(f"Number of detections: {len(results.boxes)}")
                print(f"Confidence scores: {[f'{conf:.4f}' for conf in results.boxes.conf]}")
                
        except Exception as e:
            print(f"Error processing image: {str(e)}")
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