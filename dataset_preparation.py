import os
import shutil
from pathlib import Path
import random

def split_and_organize_pallet_data(base_path):
    # Define paths
    base_path = Path("/home/uthira/pallet-detection")
    pallets_path = base_path / "data" / "Pallets"
    labels_path = base_path / "data" / "segment_labels"
    annotations_path = base_path / "data" / "Annotated_Pallets"
    dataset_path = base_path / "data" / "dataset-segment"

    # Create directories if they don't exist
    for split in ['train', 'test', 'val']:
        for subdir in ['images', 'labels', 'annotations']:
            os.makedirs(dataset_path / split / subdir, exist_ok=True)

    # Get all valid pallet images
    all_pallet_images = []
    for i in range(1, 520):
        pallet_img = pallets_path / f"pallet_{i}.jpg"
        label_file = labels_path / f"pallet_{i}.txt"
        annotation_file = annotations_path / f"detected_pallet_{i}.jpg"
        
        if pallet_img.exists() and label_file.exists() and annotation_file.exists():
            all_pallet_images.append(i)

    # Shuffle the list
    random.shuffle(all_pallet_images)
    
    # Calculate split sizes
    total_images = len(all_pallet_images)
    train_size = int(0.7 * total_images)
    val_size = int(0.15 * total_images)
    
    # Split the data
    train_images = all_pallet_images[:train_size]
    val_images = all_pallet_images[train_size:train_size + val_size]
    test_images = all_pallet_images[train_size + val_size:]

    # Create a dictionary mapping split names to their respective image lists
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }

    # Process each split
    for split_name, image_numbers in splits.items():
        split_path = dataset_path / split_name
        print(f"\nProcessing {split_name} split ({len(image_numbers)} images)...")

        for img_num in image_numbers:
            # Define source paths
            pallet_img = pallets_path / f"pallet_{img_num}.jpg"
            label_file = labels_path / f"pallet_{img_num}.txt"
            annotation_file = annotations_path / f"detected_pallet_{img_num}.jpg"

            # Define destination paths
            dest_img = split_path / "images" / f"pallet_{img_num}.jpg"
            dest_label = split_path / "labels" / f"pallet_{img_num}.txt"
            dest_annotation = split_path / "annotations" / f"detected_pallet_{img_num}.jpg"

            # Copy files
            shutil.copy2(pallet_img, dest_img)
            shutil.copy2(label_file, dest_label)
            shutil.copy2(annotation_file, dest_annotation)
            
            print(f"Processed pallet_{img_num}")

    # Print summary
    print("\nData split summary:")
    print(f"Total images: {total_images}")
    print(f"Train split: {len(train_images)} images ({len(train_images)/total_images*100:.1f}%)")
    print(f"Validation split: {len(val_images)} images ({len(val_images)/total_images*100:.1f}%)")
    print(f"Test split: {len(test_images)} images ({len(test_images)/total_images*100:.1f}%)")

if __name__ == "__main__":
    base_path = "pallet-detection"
    split_and_organize_pallet_data(base_path)