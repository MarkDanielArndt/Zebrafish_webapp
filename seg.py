from seg_helper import load_images_from_path, segment_fish, fill_holes, grow_mask
import os
import cv2
import numpy as np
import torch
from segmentation_models_pytorch import Unet
import matplotlib.pyplot as plt
from length import get_fish_length, classification_curvature

seg_directory = "runs/Segmentation"
target_size = (256, 256)

def segmentation_pipeline(folder_path):
    """
    Perform segmentation on all images in the specified folder using the provided model.
    
    Parameters:
        folder_path (str): Path to the folder containing images.
        model: Pre-trained Unet model for segmentation.
        
    Returns:
        list: A list of segmented images.
    """
    images = load_images_from_path(folder_path)
    segmented_images = []
    grown_images = []

    loaded_model = Unet(encoder_name="vgg16", encoder_weights="imagenet", in_channels=3, classes=1)
    # loaded_model = Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)
    print("Loading model...")
    if os.path.exists(f"{seg_directory}/best_model_5.pth"):
        loaded_model.load_state_dict(torch.load(f"{seg_directory}/best_model_5.pth", map_location=torch.device('cpu')))
        # loaded_model.load_state_dict(torch.load(f"{seg_directory}/segmentation_model_003.pth", map_location=torch.device('cpu')))
        loaded_model.eval()  # Set model to evaluation mode
        print(f"Model loaded from {seg_directory}/best_model_5.pth")
    else:
        print(f"Model not found at {seg_directory}/best_model_5.pth")
    # eyes_model = Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)
    # if os.path.exists(f"{seg_directory}/eye_segmentation_model.pth"):
    #     eyes_model.load_state_dict(torch.load(f"{seg_directory}/eye_segmentation_model.pth", map_location=torch.device('cpu')))
    #     eyes_model.eval()  # Set model to evaluation mode
    #     print(f"Model loaded from {seg_directory}/eye_segmentation_model.pth")

        # Define preprocessing parameters
    mean = np.array([0.485, 0.456, 0.406])  # Normalization mean
    std = np.array([0.229, 0.224, 0.225])  # Normalization std

    # Initialize a new list to store the confidence maps
    original_images = []
    #eyes_images = []

    for img in images:
        # Resize image
        original_image = np.array(img)
        #img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        # Normalize image
        processed_image = (img / 255.0 - mean) / std

        # Convert to PyTorch tensor (C, H, W) format
        input_image = torch.tensor(processed_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        
        # Get segmentation result from segment_fish function
        segmented_mask, confidence_map = segment_fish(input_image, loaded_model)
        #segmented_eyes,_ = segment_fish(input_image, eyes_model)

        # Convert the PIL image to a NumPy array
        segmented_mask_array = np.array(segmented_mask)
        
        
        # Fill holes in the segmented image
        filled_image = fill_holes(segmented_mask_array)

        # Grow the mask to include more area around the fish
        grown_image = grow_mask(filled_image)

        grown_images.append(grown_image)
        segmented_images.append(filled_image)
        original_images.append(original_image)
        #eyes_images.append(segmented_eyes)

    return original_images, segmented_images, grown_images

# original_images, segmented_images, grown_images = segmentation_pipeline(r"C:\Users\ma405l\Documents\Heidelberg_Schweregrad\Full_data\Test_data")


# for i,segmented_image in enumerate(segmented_images):
#     print(segmented_image.size)
#     masked_image, curvature = classification_curvature(original_images[i], grown_images[i])
#     fish_length = get_fish_length(segmented_image)

#     # Display the masked image
#     print(curvature)
#     plt.figure(figsize=(5, 5))
#     plt.title(f"Masked Image {i+1}")
#     plt.imshow(masked_image)
#     plt.axis('off')
#     plt.show()

# # Display original, segmented, and grown images side by side
# for i, (original_image, segmented_image, grown_image) in enumerate(zip(original_images, segmented_images, grown_images)):
#     plt.figure(figsize=(15, 5))
    
#     # Original image
#     plt.subplot(1, 3, 1)
#     plt.title(f"Original Image {i+1}")
#     plt.imshow(original_image)
#     plt.axis('off')
    
#     # Segmented image
#     plt.subplot(1, 3, 2)
#     plt.title(f"Segmented Image {i+1}")
#     plt.imshow(segmented_image, cmap='gray')
#     plt.axis('off')
    
#     # Grown image
#     plt.subplot(1, 3, 3)
#     plt.title(f"Grown Image {i+1}")
#     plt.imshow(grown_image, cmap='gray')
#     plt.axis('off')
    
#     plt.show()