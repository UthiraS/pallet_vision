import os
import torch
import wandb
from ultralytics import YOLO
import nvidia_smi

def check_gpu():
    """
    Check GPU availability and memory
    """
    if not torch.cuda.is_available():
        raise SystemError("No GPU available! This script requires GPU.")
    
    # Initialize nvidia-smi
    nvidia_smi.nvmlInit()
    device_count = nvidia_smi.nvmlDeviceGetCount()
    
    for i in range(device_count):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        
        print(f"\nGPU {i}: {nvidia_smi.nvmlDeviceGetName(handle).decode()}")
        print(f"Total memory: {info.total / 1024**2:.0f} MB")
        print(f"Free memory: {info.free / 1024**2:.0f} MB")
        print(f"Used memory: {info.used / 1024**2:.0f} MB")
    
    return device_count

def setup_wandb(device_count):
    """
    Initialize Weights & Biases with GPU info
    """
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
    
    wandb.init(
        project="pallet-detection",
        name="yolov11m-training-gpu",
        config={
            "model": "yolov11m",
            "epochs": 125,
            "batch_size": 16,
            "image_size": 640,
            "learning_rate": 0.01,
            "gpu_count": device_count,
            "gpu_name": gpu_name,
            "gpu_memory": f"{gpu_memory:.0f}MB",
            "cuda_version": torch.version.cuda
        }
    )

def train_yolo(data_path, device_count):
    """
    Train YOLOv11 model with WandB logging on GPU
    """
    # Initialize wandb
    setup_wandb(device_count)
    
    # Create data.yaml path
    yaml_path = os.path.join(data_path, 'data.yaml')
    
    # Calculate optimal batch size based on GPU memory
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    available_memory = info.free / 1024**2  # Convert to MB
    
    # Adjust batch size based on available memory (rough estimation)
    batch_size = min(16, int(available_memory / 1000))  # Assume 1GB per 16 images
    batch_size = max(4, batch_size)  # Minimum batch size of 4
    
    try:
        # Initialize YOLO model on GPU
        model = YOLO('/home/uthira/pallet-detection/weights/yolo11m.pt')
        
        # Train the model
        results = model.train(
                data=yaml_path,
                epochs=125,
                imgsz=640,
                batch=batch_size,
                device=0,  # Use first GPU
                plots=True,
                project='runs/detect',
                name='train',
                exist_ok=True,
                pretrained=True,
                optimizer='SGD',  # SGD optimizer
                lr0=0.001,  # Initial learning rate
                lrf=0.0001,  # Final learning rate factor
                momentum=0.937,  # SGD momentum/beta1
                weight_decay=0.0005,  # Optimizer weight decay
                warmup_epochs=3.0,  # Warmup epochs
                warmup_momentum=0.8,  # Warmup initial momentum
                box=7.5,  # Box loss gain
                cls=0.5,  # Classification loss gain
                dfl=1.5,  # DFL loss gain
                hsv_h=0.015,  # HSV-Hue augmentation
                hsv_s=0.7,  # HSV-Saturation augmentation
                hsv_v=0.4,  # HSV-Value augmentation
                degrees=0.0,  # Rotation augmentation
                translate=0.1,  # Translation augmentation
                scale=0.5,  # Scale augmentation
                shear=0.0,  # Shear augmentation
                perspective=0.0,  # Perspective augmentation
                flipud=0.0,  # Vertical flip augmentation
                fliplr=0.5,  # Horizontal flip augmentation
                mosaic=1.0,  # Mosaic augmentation
                mixup=0.0,  # Mixup augmentation
                copy_paste=0.0  # Copy-paste augmentation
            )

        # Log final metrics
        wandb.log({
            "final/precision": results.results_dict['metrics/precision(B)'],
            "final/recall": results.results_dict['metrics/recall(B)'],
            "final/mAP50": results.results_dict['metrics/mAP50(B)'],
            "final/mAP50-95": results.results_dict['metrics/mAP50-95(B)']
        })
        
        # Save model artifacts
        model_artifact = wandb.Artifact(
            name="yolov11m-pallet-detector",
            type="model",
            description="YOLOv11m model trained on pallet dataset"
        )
        model_artifact.add_file(f"/home/uthira/pallet-detection/weights/yolo11m.pt")
        wandb.log_artifact(model_artifact)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        wandb.finish()
    
    finally:
        wandb.finish()
        nvidia_smi.nvmlShutdown()

def main():
    print("Starting YOLOv11 training pipeline with GPU and WandB logging...")
    
    # Install required packages
    os.system("pip install ultralytics supervision wandb nvidia-ml-py3")
    
    # Check GPU and get device count
    device_count = check_gpu()
    
    # Set your dataset path
    data_path = "/home/uthira/pallet-detection/data/dataset"  # Update this to your dataset path
    
    # Login to WandB
    wandb.login()
    
    # Train model
    print("\nTraining YOLOv11 model on GPU...")
    train_yolo(data_path, device_count)
    
    print("\nTraining pipeline completed!")
    print(f"Results can be found in: {os.getcwd()}/runs/detect/")
    print("Check WandB dashboard for detailed metrics and visualizations")

if __name__ == "__main__":
    main()