import os
import numpy as np
import cv2
from typing import Tuple, Optional
from PIL import Image
from tqdm import tqdm

class MedicalImageAugmentor:
    """
    Medical image augmentation focusing on spatial transformations.
    Designed for 512x512 grayscale images with 13-class segmentation labels (0-12).
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)

    def rotate_and_flip(self, 
                       image: np.ndarray, 
                       label: np.ndarray, 
                       max_angle: float = 15) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augmentation 1: Rotation + Random Flip
        Applies a small rotation followed by random flip
        """
        # Rotation
        angle = np.random.uniform(-max_angle, max_angle)
        height, width = image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        image_rotated = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        label_rotated = cv2.warpAffine(
            label, 
            rotation_matrix, 
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT
        )
        
        # # Random flip (horizontal or vertical)
        # if np.random.random() > 0.5:
        #     image_rotated = np.fliplr(image_rotated)
        #     label_rotated = np.fliplr(label_rotated)
        # elif np.random.random() > 0.5:
        #     image_rotated = np.flipud(image_rotated)
        #     label_rotated = np.flipud(label_rotated)
            
        return image_rotated, label_rotated

    def crop_and_resize(self,
                       image: np.ndarray,
                       label: np.ndarray,
                       min_crop_ratio: float = 0.9,
                       max_crop_ratio: float = 0.99) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augmentation 2: Random Crop and Resize
        Crops a random region and resizes it back to original size
        """
        h, w = image.shape
        min_size = int(min(h, w) * min_crop_ratio)
        max_size = int(min(h, w) * max_crop_ratio)
        crop_size = np.random.randint(min_size, max_size + 1)
        
        # Random crop coordinates
        max_top = h - crop_size
        max_left = w - crop_size
        top = np.random.randint(0, max_top + 1)
        left = np.random.randint(0, max_left + 1)
        
        # Perform crop
        image_crop = image[top:top+crop_size, left:left+crop_size]
        label_crop = label[top:top+crop_size, left:left+crop_size]
        
        # Resize back to original size
        image_resized = cv2.resize(image_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        label_resized = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return image_resized, label_resized

def process_folders(train_images_dir: str, train_labels_dir: str):
    """
    Process all images in the training folders and create 2 augmented versions
    using different spatial transformations
    """
    augmentor = MedicalImageAugmentor(seed=42)
    
    # Get list of subfolders
    image_subfolders = [f for f in os.listdir(train_images_dir) 
                       if os.path.isdir(os.path.join(train_images_dir, f))]
    
    print(f"Found {len(image_subfolders)} subfolders to process")
    
    for subfolder in image_subfolders:
        print(f"\nProcessing subfolder: {subfolder}")
        
        img_subfolder_path = os.path.join(train_images_dir, subfolder)
        label_subfolder_path = os.path.join(train_labels_dir, subfolder)
        
        # Get all PNG files in the subfolder
        png_files = [f for f in os.listdir(img_subfolder_path) if f.endswith('.png') and '_' not in f]
        
        # Process each image in the subfolder
        for png_file in tqdm(png_files, desc=f"Processing {subfolder}"):
            try:
                # Load image and label
                image_path = os.path.join(img_subfolder_path, png_file)
                label_path = os.path.join(label_subfolder_path, png_file)
                
                image = np.array(Image.open(image_path).convert('L'))
                label = np.array(Image.open(label_path))
                
                # Create two augmented versions
                # 1. Rotation and flip
                aug_img_1, aug_label_1 = augmentor.rotate_and_flip(image, label)
                
                # Save first augmentation
                base_name = os.path.splitext(png_file)[0]
                aug_image_name = f"{base_name}_rotate_flip.png"
                aug_label_name = f"{base_name}_rotate_flip.png"
                
                Image.fromarray(aug_img_1).save(os.path.join(img_subfolder_path, aug_image_name))
                Image.fromarray(aug_label_1).save(os.path.join(label_subfolder_path, aug_label_name))
                
                # 2. Crop and resize
                aug_img_2, aug_label_2 = augmentor.crop_and_resize(image, label)
                
                # Save second augmentation
                aug_image_name = f"{base_name}_crop_resize.png"
                aug_label_name = f"{base_name}_crop_resize.png"
                
                Image.fromarray(aug_img_2).save(os.path.join(img_subfolder_path, aug_image_name))
                Image.fromarray(aug_label_2).save(os.path.join(label_subfolder_path, aug_label_name))
                
            except Exception as e:
                print(f"Error processing {png_file}: {str(e)}")
                continue

def main():
    """
    Main function to run the batch augmentation process
    """
    # Set your directories here
    train_images_dir = "data/Public_leaderboard_data/train_images"  # Replace with your image directory path
    train_labels_dir = "data/Public_leaderboard_data/train_labels"  # Replace with your label directory path
    
    try:
        # Verify directories exist
        if not os.path.exists(train_images_dir):
            raise ValueError(f"Image directory {train_images_dir} does not exist")
        if not os.path.exists(train_labels_dir):
            raise ValueError(f"Label directory {train_labels_dir} does not exist")
            
        print("Starting batch augmentation process...")
        process_folders(train_images_dir, train_labels_dir)
        print("\nAugmentation completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
