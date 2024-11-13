import os
from pathlib import Path
import cv2

def normalize_labels(dataset_path):
    """
    Normalize YOLO format labels based on image dimensions
    """
    for split in ['train', 'val', 'test']:
        images_path = Path(dataset_path) / split / 'images'
        labels_path = Path(dataset_path) / split / 'labels'
        
        # Create labels directory if it doesn't exist
        labels_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {split} split...")
        for img_path in images_path.glob('*.jpg'):
            try:
                # Read image to get dimensions
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Could not read {img_path}")
                    continue
                    
                height, width = img.shape[:2]
                
                # Get corresponding label file
                label_file = labels_path / f"{img_path.stem}.txt"
                if not label_file.exists():
                    print(f"No label file for {img_path.name}")
                    continue
                
                # Read and normalize labels
                normalized_lines = []
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        try:
                            parts = line.strip().split()
                            if len(parts) == 5:  # class x y w h format
                                class_id = parts[0]
                                x = float(parts[1])
                                y = float(parts[2])
                                w = float(parts[3])
                                h = float(parts[4])
                                
                                # Normalize coordinates
                                x_norm = x / width
                                y_norm = y / height
                                w_norm = w / width
                                h_norm = h / height
                                
                                # Check if normalized values are valid
                                if 0 <= x_norm <= 1 and 0 <= y_norm <= 1 and 0 <= w_norm <= 1 and 0 <= h_norm <= 1:
                                    normalized_lines.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                                else:
                                    print(f"Invalid normalized coordinates in {label_file}: {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
                        except Exception as e:
                            print(f"Error processing line in {label_file}: {line.strip()}")
                            continue
                
                # Write normalized labels
                if normalized_lines:
                    with open(label_file, 'w') as f:
                        f.writelines(normalized_lines)
                    print(f"Processed: {img_path.name} - {len(normalized_lines)} labels")
                else:
                    print(f"No valid labels for {img_path.name}")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

if __name__ == "__main__":
    dataset_path = "/home/uthira/pallet-detection/data/dataset"  # Update this to your dataset path
    normalize_labels(dataset_path)
    
    print("\nLabel normalization completed!")
    print("You can now run YOLOv11 training.")