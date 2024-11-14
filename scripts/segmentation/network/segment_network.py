import os
import torch
import wandb
from ultralytics import YOLO
import nvidia_smi

torch.cuda.empty_cache()
# Enable deterministic behavior
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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
        project="pallet-segmentation",
        name="yolo11m-seg-training",
        config={
            "model": "yolo11s-seg",
            "epochs": 25,
            "batch_size": 8,  # Reduced batch size for segmentation
            "image_size": 640,
            "learning_rate": 0.001,
            "gpu_count": device_count,
            "gpu_name": gpu_name,
            "gpu_memory": f"{gpu_memory:.0f}MB",
            "cuda_version": torch.version.cuda,
            "task": "segment"
        }
    )

def train_yolo_seg(data_path, device_count):
    """
    Train YOLOv11 model for instance segmentation with WandB logging
    """
    # Initialize wandb
    setup_wandb(device_count)
    
    # Create data.yaml path
    yaml_path = os.path.join(data_path, 'data.yaml')
    
    # Calculate optimal batch size based on GPU memory
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    available_memory = info.free / 1024**2
    
    # Reduced batch size for segmentation due to higher memory requirements
    batch_size = 8
    
    try:
        # Initialize YOLO segmentation model
        model = YOLO('/home/uthira/pallet-detection/weights/yolo11s-seg.pt')
        
        # Train the model with segmentation-specific parameters
        results = model.train(
                data=yaml_path,
                task='segment',
                epochs=225,
                imgsz=640,
                batch=batch_size,
                device=0,
                plots=True,
                project='runs/segment',
                name='train',
                exist_ok=True,
                pretrained=True,
                optimizer='AdamW',
                lr0=0.001,
                lrf=0.0001,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3.0,
                warmup_momentum=0.8,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.0,
                copy_paste=0.0,
                cache=False,
                rect=False,
                cos_lr=True,
                close_mosaic=10,
                amp=True
            )

        # Log final metrics
        wandb.log({
            "final/precision": results.results_dict['metrics/precision(B)'],
            "final/recall": results.results_dict['metrics/recall(B)'],
            "final/mAP50": results.results_dict['metrics/mAP50(B)'],
            "final/mAP50-95": results.results_dict['metrics/mAP50-95(B)'],
            "final/mask_mAP50": results.results_dict.get('metrics/mask_mAP50(B)', 0),
            "final/mask_mAP50-95": results.results_dict.get('metrics/mask_mAP50-95(B)', 0)
        })
        
        # Save model artifacts
        model_artifact = wandb.Artifact(
            name="yolo11m-seg-pallet-detector",
            type="model",
            description="YOLOv11m segmentation model trained on pallet dataset"
        )
        model_artifact.add_file(os.path.join(os.getcwd(), 'runs/segment/train/weights/best.pt'))
        wandb.log_artifact(model_artifact)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        wandb.finish()
    
    finally:
        wandb.finish()
        nvidia_smi.nvmlShutdown()

def main():
    print("Starting YOLOv11 Segmentation training pipeline with GPU and WandB logging...")
    
    # Install required packages
    os.system("pip install ultralytics supervision wandb nvidia-ml-py3")
    
    # Check GPU and get device count
    device_count = check_gpu()
    
    # Set your dataset path
    data_path = "/home/uthira/pallet-detection/data/dataset-segment"  # Update this path
    
    # Login to WandB
    wandb.login()
    
    # Train model
    print("\nTraining YOLOv11 segmentation model on GPU...")
    train_yolo_seg(data_path, device_count)
    
    print("\nTraining pipeline completed!")
    print(f"Results can be found in: {os.getcwd()}/runs/segment/")
    print("Check WandB dashboard for detailed metrics and visualizations")

if __name__ == "__main__":
    main()