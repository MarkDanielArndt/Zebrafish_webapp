from seg_helper import load_images_from_path, segment_fish, fill_holes, grow_mask
import os
import cv2
import numpy as np
import torch
from segmentation_models_pytorch import Unet
from huggingface_hub import hf_hub_download

target_size = (256, 256)

def _load_unet_model(model_path=None, repo_id=None, filename=None, label="model"):
    """
    Load a binary Unet model from a local path or from Hugging Face Hub.
    Returns the model instance when successful, otherwise None.
    """
    model = Unet(encoder_name="vgg16", encoder_weights="imagenet", in_channels=3, classes=1)
    resolved_path = None

    if model_path and os.path.exists(model_path):
        resolved_path = model_path
    elif repo_id and filename:
        try:
            resolved_path = hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as exc:
            print(f"Could not download {label} from Hugging Face Hub: {exc}")
            return None
    elif filename and os.path.exists(filename):
        resolved_path = filename

    if not resolved_path:
        print(f"{label.capitalize()} not found.")
        return None

    try:
        model.load_state_dict(torch.load(resolved_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"{label.capitalize()} loaded from {resolved_path}")
        return model
    except Exception as exc:
        print(f"Failed to load {label} from {resolved_path}: {exc}")
        return None


def segmentation_pipeline(
    folder_path,
    include_eyes=False,
    body_repo_id="markdanielarndt/Zebrafish_Segmentation",
    body_model_filename="best_model_5.pth",
    eye_model_path=None,
    eye_repo_id="markdanielarndt/Zebrafish_Segmentation",
    eye_model_filename="best_model_eyes_combined_230226.pth",
):
    """
    Perform body segmentation on all images in the specified folder.

    Optional eye segmentation can be enabled by setting include_eyes=True.

    Returns:
        - default: (original_images, segmented_images, grown_images)
        - if include_eyes=True: (original_images, segmented_images, grown_images, eyes_images)
    """
    images = load_images_from_path(folder_path)
    segmented_images = []
    grown_images = []
    original_images = []
    eyes_images = []

    print("Loading body segmentation model...")
    loaded_model = _load_unet_model(
        repo_id=body_repo_id,
        filename=body_model_filename,
        label="body model",
    )

    if loaded_model is None:
        raise RuntimeError("Body segmentation model could not be loaded.")

    eyes_model = None
    if include_eyes:
        print(f"Loading eye segmentation model from {eye_repo_id}/{eye_model_filename}...")
        eyes_model = _load_unet_model(
            model_path=eye_model_path,
            repo_id=eye_repo_id,
            filename=eye_model_filename,
            label="eye model",
        )
        if eyes_model is None:
            print(f"WARNING: Eye model unavailable at {eye_repo_id}/{eye_model_filename}. Returning empty eye masks.")
        else:
            print("Eye model loaded successfully!")

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

        if include_eyes:
            if eyes_model is not None:
                segmented_eyes, _ = segment_fish(input_image, eyes_model, biggest_only=True)
                segmented_eyes_array = np.array(segmented_eyes)
            else:
                segmented_eyes_array = np.zeros(target_size, dtype=np.uint8)
            eyes_images.append(segmented_eyes_array)

        grown_images.append(grown_image)
        segmented_images.append(filled_image)
        original_images.append(original_image)

    if include_eyes:
        return original_images, segmented_images, grown_images, eyes_images

    return original_images, segmented_images, grown_images
