import numpy as np
import cv2
from typing import Tuple, List, Optional
from PIL import Image
import os

class MedicalImageAugmentor:
    """
    A class for medical image augmentation with minimal dependencies.
    Specifically designed for 512x512 grayscale images with 13-class segmentation labels (0-12).
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the augmentor.
        
        Args:
            seed (int, optional): Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

    def rotate(self, 
              image: np.ndarray, 
              label: np.ndarray, 
              max_angle: float = 15) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate image and label by a random angle within [-max_angle, max_angle].
        Uses bilinear interpolation for image and nearest neighbor for label to preserve class values.
        
        Args:
            image: Input image array (512, 512)
            label: Segmentation label array (512, 512)
            max_angle: Maximum rotation angle in degrees
            
        Returns:
            Tuple of rotated image and label arrays
        """
        angle = np.random.uniform(-max_angle, max_angle)
        height, width = image.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image using bilinear interpolation
        image_rotated = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        # Rotate label using nearest neighbor interpolation to preserve label values
        label_rotated = cv2.warpAffine(
            label, 
            rotation_matrix, 
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return image_rotated, label_rotated

    def adjust_contrast(self,
                       image: np.ndarray,
                       label: np.ndarray,
                       factor_range: Tuple[float, float] = (0.8, 1.2)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adjust image contrast while keeping labels unchanged.
        
        Args:
            image: Input image array
            label: Segmentation label array
            factor_range: Tuple of (min_factor, max_factor) for contrast adjustment
            
        Returns:
            Tuple of contrast-adjusted image and original label
        """
        factor = np.random.uniform(factor_range[0], factor_range[1])
        mean = np.mean(image)
        adjusted = mean + factor * (image - mean)
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted, label

    def add_gaussian_noise(self,
                          image: np.ndarray,
                          label: np.ndarray,
                          std_range: Tuple[float, float] = (5, 20)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add Gaussian noise to the image while keeping labels unchanged.
        
        Args:
            image: Input image array
            label: Segmentation label array
            std_range: Range of standard deviation for Gaussian noise
            
        Returns:
            Tuple of noisy image and original label
        """
        std = np.random.uniform(std_range[0], std_range[1])
        noise = np.random.normal(0, std, image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image, label

    def random_flip(self,
                   image: np.ndarray,
                   label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly flip image and label horizontally and/or vertically.
        
        Args:
            image: Input image array
            label: Segmentation label array
            
        Returns:
            Tuple of flipped image and label arrays
        """
        # # Horizontal flip
        # if np.random.random() > 0.5:
        #     image = np.fliplr(image)
        #     label = np.fliplr(label)
            
        # # Vertical flip    
        # if np.random.random() > 0.5:
        #     image = np.flipud(image)
        #     label = np.flipud(label)
            
        return image, label

    def random_crop_resize(self,
                          image: np.ndarray,
                          label: np.ndarray,
                          min_crop_ratio: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly crop and resize back to original size.
        
        Args:
            image: Input image array
            label: Segmentation label array
            min_crop_ratio: Minimum ratio of crop size to original size
            
        Returns:
            Tuple of cropped and resized image and label arrays
        """
        h, w = image.shape
        min_size = int(min(h, w) * min_crop_ratio)
        crop_size = np.random.randint(min_size, min(h, w))
        
        # Random crop coordinates
        top = np.random.randint(0, h - crop_size + 1)
        left = np.random.randint(0, w - crop_size + 1)
        
        # Crop
        image_crop = image[top:top+crop_size, left:left+crop_size]
        label_crop = label[top:top+crop_size, left:left+crop_size]
        
        # Resize back to original size
        image_resized = cv2.resize(image_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        label_resized = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return image_resized, label_resized

    def apply_augmentations(self,
                           image: np.ndarray,
                           label: np.ndarray,
                           augment_types: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a sequence of augmentations to the image and label.
        
        Args:
            image: Input image array
            label: Segmentation label array
            augment_types: List of augmentation types to apply
                         Options: ['rotate', 'contrast', 'noise', 'flip', 'crop']
            
        Returns:
            Tuple of augmented image and label arrays
        """
        # Create augmentation mapping
        aug_functions = {
            'rotate': self.rotate,
            'contrast': self.adjust_contrast,
            'noise': self.add_gaussian_noise,
            'flip': self.random_flip,
            'crop': self.random_crop_resize
        }
        
        # Validate augmentation types
        valid_types = set(aug_functions.keys())
        for aug_type in augment_types:
            if aug_type not in valid_types:
                raise ValueError(f"Invalid augmentation type: {aug_type}. "
                              f"Valid types are: {valid_types}")
        
        # Apply selected augmentations sequentially
        for aug_type in augment_types:
            image, label = aug_functions[aug_type](image, label)
            
        return image, label

def verify_label_integrity(label: np.ndarray) -> bool:
    """
    Verify that the label array contains only valid classes (0-12).
    
    Args:
        label: Segmentation label array
        
    Returns:
        bool: True if label contains only valid classes
    """
    unique_classes = np.unique(label)
    return all(cls in range(13) for cls in unique_classes)

def main():
    """
    Example usage of the MedicalImageAugmentor class.
    """
    # Load example image and label
    image_path = "example_image.png"
    label_path = "example_label.png"
    
    try:
        # Load and verify input data
        image = np.array(Image.open(image_path).convert('L'))
        label = np.array(Image.open(label_path))
        
        if image.shape != (512, 512) or label.shape != (512, 512):
            raise ValueError("Image and label must be 512x512")
        
        if not verify_label_integrity(label):
            raise ValueError("Label contains invalid classes (should be 0-12)")
        
        # Create augmentor instance with fixed seed for reproducibility
        augmentor = MedicalImageAugmentor(seed=42)
        
        # Example 1: Single augmentation
        print("Applying rotation augmentation...")
        img_rotated, label_rotated = augmentor.rotate(image, label)
        Image.fromarray(img_rotated).save('rotated_image.png')
        Image.fromarray(label_rotated).save('rotated_label.png')
        
        # Example 2: Multiple augmentations
        print("Applying multiple augmentations...")
        augmentation_sequence = ['flip', 'contrast', 'noise']
        img_multi_aug, label_multi_aug = augmentor.apply_augmentations(
            image,
            label,
            augment_types=augmentation_sequence
        )
        Image.fromarray(img_multi_aug).save('multi_augmented_image.png')
        Image.fromarray(label_multi_aug).save('multi_augmented_label.png')
        
        # Example 3: Custom augmentation sequence
        print("Applying custom augmentation sequence...")
        custom_sequence = ['rotate', 'flip', 'crop']
        img_custom, label_custom = augmentor.apply_augmentations(
            image,
            label,
            augment_types=custom_sequence
        )
        Image.fromarray(img_custom).save('custom_augmented_image.png')
        Image.fromarray(label_custom).save('custom_augmented_label.png')
        
        print("Augmentation completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    train_path = 'data/Public_leaderboard_data/train_images_clean'
    label_path = 'data/Public_leaderboard_data/train_labels'
    new_train_path = 'data/Augmented/train_images_clean'
    new_label_path = 'data/Augmented/train_labels'

    augmentor = MedicalImageAugmentor(seed=42)
    custom_sequence = ['rotate', 'flip', 'crop']
    
    for folder in os.listdir(train_path):
        os.makedirs(os.path.join(new_train_path, folder), exist_ok=True)
        os.makedirs(os.path.join(new_label_path, folder), exist_ok=True)

        for img in os.listdir(os.path.join(train_path, folder)):
            image = np.array(Image.open(os.path.join(train_path, folder, img)).convert('L'))
            label = np.array(Image.open(os.path.join(label_path, folder, img)))

            img_custom, label_custom = augmentor.apply_augmentations(
                image,
                label,
                augment_types=custom_sequence
            )

            Image.fromarray(img_custom).save('custom_augmented_image.png')
            Image.fromarray(label_custom).save('custom_augmented_label.png')
