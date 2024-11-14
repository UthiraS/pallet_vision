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
        """
        Initialize inference
        Args:
            model_path: Path to model weights
        """
        self.model = YOLO(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    # def display_results(self, original_image, pred_masks, conf_scores):
    #     """
    #     Display inference results using cv2.imshow without blocking
    #     """
    #     # Create visualization with original and predictions side by side
    #     h, w = original_image.shape[:2]
    #     grid = np.zeros((h, w*2, 3), dtype=np.uint8)
        
    #     # Original image (left)
    #     grid[:h, :w] = original_image
        
    #     # Predictions overlay (right)
    #     pred_vis = original_image.copy()
    #     colors = np.random.randint(0, 255, size=(len(pred_masks), 3), dtype=np.uint8)
        
    #     for i, (mask, conf) in enumerate(zip(pred_masks, conf_scores)):
    #         color = colors[i].tolist()
            
    #         # Create mask overlay
    #         overlay = np.zeros_like(original_image)
    #         overlay[mask] = color
    #         pred_vis = cv2.addWeighted(pred_vis, 0.7, overlay, 0.3, 0)
            
    #         # Add confidence score text
    #         y_pos = 30 + (i * 30)
    #         text = f"Instance {i+1}, Conf: {conf:.2f}"
    #         cv2.putText(pred_vis, text, (10, y_pos), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    #     grid[:h, w:] = pred_vis
        
    #     # Display the results with a 1ms wait instead of waiting for key
    #     cv2.imshow('Segmentation Results', grid)
    #     cv2.waitKey(1)  # Changed from waitKey(0) to waitKey(1)

    # def run_inference(self, image):
    #     """
    #     Run inference on single image
    #     """
    #     # Load image
    #     # image = cv2.imread(image_path)

    #     # if image is None:
    #     #     print(f"Failed to load image: {image_path}")
    #     #     return None
            
    #     try:
    #         # Run inference
    #         results = self.model.predict(
    #             source=image,
    #             task='segment',
    #             conf=0.25,
    #             iou=0.7,
    #             device=self.device
    #         )[0]
            
    #         if results.masks is not None:
    #             # Get masks and confidence scores
    #             pred_masks = results.masks.data.cpu().numpy()
    #             conf_scores = results.boxes.conf.cpu().numpy()
                
    #             # Resize masks to match image dimensions
    #             restored_masks = []
    #             for mask in pred_masks:
    #                 if mask.shape[:2] != image.shape[:2]:
    #                     mask = cv2.resize(mask.astype(float), 
    #                                    (image.shape[1], image.shape[0])) > 0.5
    #                 restored_masks.append(mask)
                
    #             # Display results
    #             self.display_results(image, restored_masks, conf_scores)
                
    #             # Print results
    #             print(f"\nResults:")
    #             print(f"Number of instances detected: {len(restored_masks)}")
    #             print(f"Confidence scores: {[f'{conf:.4f}' for conf in conf_scores]}")
                
    #         else:
    #             print("No detections found in the image")
    #             # Display original image if no detections
    #             cv2.imshow('No Detections', image)
    #             cv2.waitKey(1)  # Changed from waitKey(0) to waitKey(1)
    #             # Removed destroyAllWindows call
                
    #     except Exception as e:
    #         print(f"Error processing image: {str(e)}")
    #         import traceback
    #         traceback.print_exc()

    def display_results(self, original_image, pred_masks=None, conf_scores=None):
        """
        Display inference results using cv2.imshow with better visibility
        """
        # Create visualization with original and predictions side by side
        h, w = original_image.shape[:2]
        grid = np.zeros((h + 30, w*2, 3), dtype=np.uint8)
        
        # Original image (left)
        grid[0:h, :w] = original_image
        
        # Add caption for original image
        cv2.putText(grid, 'Original Image', (10, h + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Right side - either segmentation or original
        if pred_masks is not None and len(pred_masks) > 0:
            # Start with original image
            pred_vis = original_image.copy()
            colors = np.random.randint(0, 255, size=(len(pred_masks), 3), dtype=np.uint8)
            
            # Create a single overlay for all masks
            combined_overlay = np.zeros_like(original_image)
            
            for i, mask in enumerate(pred_masks):
                color = colors[i].tolist()
                combined_overlay[mask] = color
            
            # Blend with higher alpha for original image
            pred_vis = cv2.addWeighted(pred_vis, 0.9, combined_overlay, 0.5, 0)
            
            grid[0:h, w:] = pred_vis
        else:
            # If no detections, show original image on right side
            grid[0:h, w:] = original_image
        
        # Add caption for segmentation
        cv2.putText(grid, 'Segmentation Output', (w + 10, h + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the results
        cv2.imshow('Pallet Detection Results', grid)
        cv2.waitKey(3000)


    def run_inference(self, image_path):
        """
        Run inference on single image
        """
        # Load image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image: {image_path}")
            return None

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
                
                # Display results
                self.display_results(image, restored_masks, conf_scores)
                
                # Print results
                print(f"\nResults:")
                print(f"Number of instances detected: {len(restored_masks)}")
                print(f"Confidence scores: {[f'{conf:.4f}' for conf in conf_scores]}")
                
            else:
                # No detections - display original image on both sides
                self.display_results(image)
                    
        except Exception as e:
            print(f"Error processing image: {str(e)}")
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