import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import shutil
from tqdm import tqdm

class PalletSegmentationTrainer:
    def __init__(self, dataset_path: Path, model_size='l'):
        """
        Initialize trainer for pallet segmentation
        
        Args:
            dataset_path: Path to dataset root
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        """
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        self.data_yaml = None
        self.model = None
    
    def setup_dataset(self):
        """Create data.yaml configuration file."""
        # Verify dataset structure
        required_dirs = [
            self.dataset_path / 'train' / 'images',
            self.dataset_path / 'train' / 'labels',
            self.dataset_path / 'val' / 'images',
            self.dataset_path / 'val' / 'labels',
            self.dataset_path / 'test' / 'images',
            self.dataset_path / 'test' / 'labels'
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                raise FileNotFoundError(f"Required directory missing: {directory}")

        # Create data.yaml
        data_yaml = {
            'path': str(self.dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {
                1: 'pallet'  # Single class for pallets
            },
            'nc': 1  # Number of classes
        }
        
        yaml_path = self.dataset_path / 'data.yaml'
        # with open(yaml_path, 'w') as f:
        #     yaml.dump(data_yaml, f)
        
        self.data_yaml = yaml_path
        return yaml_path
    
    def print_dataset_info(self):
        """Print information about the dataset."""
        splits = ['train', 'val', 'test']
        print("\nDataset Statistics:")
        print("=" * 50)
        
        for split in splits:
            img_dir = self.dataset_path / split / 'images'
            label_dir = self.dataset_path / split / 'labels'
            
            n_images = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
            n_labels = len(list(label_dir.glob('*.txt')))
            
            print(f"\n{split.upper()}:")
            print(f"  Images: {n_images}")
            print(f"  Labels: {n_labels}")
            if n_images != n_labels:
                print(f"  WARNING: Mismatch between images and labels!")
    
    def setup_training(self):
        """Initialize model and setup training configuration."""
        print("\nInitializing training setup...")
        
        # Setup dataset
        self.setup_dataset()
        self.print_dataset_info()
        
        # Initialize model
        model_name = f'/home/uthira/pallet-detection/weights/yolo11{self.model_size}-seg.pt'
        print(f"\nLoading {model_name}...")
        self.model = YOLO(model_name)
        
        return self.model
    
    def train(self, epochs=100, batch_size=8, imgsz=640):
        """
        Train the model with specified parameters
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_training first.")
        
        print("\nStarting training...")
        self.model.train(
            data=str(self.data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=0,  # Use GPU
            patience=50,  # Early stopping patience
            save=True,
            plots=True,
            augment=True,
            mixup=0.3,
            mosaic=1.0,
            copy_paste=0.4,
            degrees=10.0,
            translate=0.2,
            scale=0.5,
            shear=2.0,
            perspective=0.0,
            flipud=0.5,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            cache=True,
            save_period=10,
            workers=8,
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            cos_lr=True,  # Cosine learning rate scheduling
            close_mosaic=10,  # Disable mosaic augmentation for final epochs
            resume=False
        )
    
    def validate(self):
        """Validate the trained model."""
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_training first.")
        
        print("\nRunning validation...")
        results = self.model.val(data=str(self.data_yaml))
        return results
    
    def export_model(self, format='onnx'):
        """Export the trained model to specified format."""
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_training first.")
        
        print(f"\nExporting model to {format}...")
        self.model.export(format=format)

def main():
    # Configuration
    DATASET_PATH = Path("/home/uthira/pallet-detection/data/dataset")  # Update this to your dataset path
    MODEL_SIZE = 's'  # Choose from 'n', 's', 'm', 'l', 'x'
    
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 8
    IMAGE_SIZE = 640
    
    # Initialize trainer
    trainer = PalletSegmentationTrainer(
        dataset_path=DATASET_PATH,
        model_size=MODEL_SIZE
    )
    
    try:
        # Setup training
        print("Setting up training pipeline...")
        trainer.setup_training()
        
        # Train model
        print("\nStarting training process...")
        trainer.train(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            imgsz=IMAGE_SIZE
        )
        
        # Validate model
        print("\nValidating model...")
        results = trainer.validate()
        
        print("\nTraining completed!")
        print(f"Segmentation mAP: {results.seg.map}")
        print(f"Box mAP50: {results.box.map50}")
        
        # Export model
        trainer.export_model(format='onnx')
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main()