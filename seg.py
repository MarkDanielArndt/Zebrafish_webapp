from seg_helper import load_images_from_path, segment_fish, fill_holes, grow_mask
import os
import cv2
import numpy as np
import torch
from segmentation_models_pytorch import Unet
from huggingface_hub import hf_hub_download

target_size = (256, 256)

def segmentation_pipeline(folder_path):
    """
    Perform segmentation on all images in the specified folder using the Hugging Face Hub model.
    """
    images = load_images_from_path(folder_path)
    segmented_images = []
    grown_images = []
    original_images = []

    # Build model
    loaded_model = Unet(encoder_name="vgg16", encoder_weights="imagenet", in_channels=3, classes=1)
    print("Loading segmentation model from Hugging Face Hub...")

    # Download model weights
    model_path = hf_hub_download(
        repo_id="markdanielarndt/Zebrafish_Segmentation",
        filename="best_model_5.pth"
    )

    loaded_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    loaded_model.eval()
    print("Segmentation model loaded successfully.")

    # Preprocessing parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for img in images:
        original_image = np.array(img)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        processed_image = (img / 255.0 - mean) / std
        input_image = torch.tensor(processed_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        segmented_mask, confidence_map = segment_fish(input_image, loaded_model)
        segmented_mask_array = np.array(segmented_mask)

        filled_image = fill_holes(segmented_mask_array)
        grown_image = grow_mask(filled_image)

        grown_images.append(grown_image)
        segmented_images.append(filled_image)
        original_images.append(original_image)

    return original_images, segmented_images, grown_images
