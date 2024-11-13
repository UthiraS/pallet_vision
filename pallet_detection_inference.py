import os
import torch
import wandb
from ultralytics import YOLO
import nvidia_smi
import cv2
import supervision as sv
from pathlib import Path

def validate_model(data_path):
    """
    Validate the trained model on GPU and log metrics to WandB
    """
    weights_path = f"{os.getcwd()}/runs/detect/train/weights/best.pt"
    
    try:
        # Initialize model with best weights
        model = YOLO(weights_path)
        
        # Run validation on GPU
        metrics = model.val(
            data=f"{data_path}/data.yaml",
            batch=16,
            device=0,  # Use GPU
            logger='wandb',
            split='val',  # Use validation split
            conf=0.25,  # Confidence threshold
            iou=0.7,    # NMS IoU threshold
            max_det=300,  # Maximum detections per image
            half=True,  # Use half precision
        )
        
        # Log validation metrics to WandB
        wandb.log({
            "val/precision": metrics.results_dict['metrics/precision(B)'],
            "val/recall": metrics.results_dict['metrics/recall(B)'],
            "val/mAP50": metrics.results_dict['metrics/mAP50(B)'],
            "val/mAP50-95": metrics.results_dict['metrics/mAP50-95(B)']
        })
        
        print("\nValidation Results:")
        print(f"Precision: {metrics.results_dict['metrics/precision(B)']:.4f}")
        print(f"Recall: {metrics.results_dict['metrics/recall(B)']:.4f}")
        print(f"mAP50: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"mAP50-95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")

def run_inference(data_path):
    """
    Run inference on test images using GPU and log results to WandB
    """
    weights_path = f"{os.getcwd()}/runs/detect/train/weights/best.pt"
    test_images_path = os.path.join(data_path, 'test', 'images')
    output_path = os.path.join(os.getcwd(), 'runs/detect/inference')
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Initialize model with best weights
        model = YOLO(weights_path)
        
        # Initialize supervision annotator
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # Get all test images
        test_images = [f for f in Path(test_images_path).glob('*') 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        # Create a WandB table for results
        results_table = wandb.Table(columns=["image", "predictions", "confidence"])
        
        for img_path in test_images:
            # Run inference
            results = model.predict(
                source=str(img_path),
                conf=0.25,
                iou=0.7,
                max_det=300,
                device=0,  # Use GPU
                half=True,  # Use half precision
                save=True,
                save_txt=True,
                project=output_path,
                name='results'
            )[0]
            
            # Get image for annotation
            image = cv2.imread(str(img_path))
            
            # Convert results to supervision format
            detections = sv.Detections.from_ultralytics(results)
            
            # Annotate image
            annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
            
            # Save annotated image
            output_img_path = os.path.join(output_path, 'results', f'annotated_{img_path.name}')
            cv2.imwrite(output_img_path, annotated_image)
            
            # Log results to WandB
            # Convert image to wandb Image with boxes
            boxes_media_item = wandb.Image(
                annotated_image,
                caption=f"Predictions for {img_path.name}"
            )
            
            # Add to results table
            results_table.add_data(
                boxes_media_item,
                len(results.boxes),
                results.boxes.conf.mean().item() if len(results.boxes) > 0 else 0
            )
            
            print(f"Processed {img_path.name}: Found {len(results.boxes)} objects")
        
        # Log results table to WandB
        wandb.log({"inference_results": results_table})
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")

def main():
    print("Starting YOLOv11 validation and inference pipeline...")
    
    # Install required packages
    os.system("pip install ultralytics supervision wandb nvidia-ml-py3")
    
    # Set your dataset path
    data_path = "dataset"  # Update this to your dataset path
    
    # Login to WandB
    wandb.login()
    
    # Initialize new WandB run for validation and inference
    wandb.init(
        project="pallet-detection",
        name="yolov11m-validation-inference",
        config={
            "model": "yolov11m",
            "conf_threshold": 0.25,
            "iou_threshold": 0.7,
            "gpu": torch.cuda.get_device_name(0)
        }
    )
    
    # Run validation
    print("\nRunning validation...")
    validate_model(data_path)
    
    # Run inference
    print("\nRunning inference on test images...")
    run_inference(data_path)
    
    # Finish WandB run
    wandb.finish()
    
    print("\nValidation and inference completed!")
    print(f"Results can be found in: {os.getcwd()}/runs/detect/")
    print("Check WandB dashboard for detailed metrics and visualizations")

if __name__ == "__main__":
    main()