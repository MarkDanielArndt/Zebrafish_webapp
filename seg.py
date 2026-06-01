from seg_helper import load_images_from_path, segment_fish, fill_holes, grow_mask
import os
import cv2
import numpy as np
import torch
from segmentation_models_pytorch import Unet
from huggingface_hub import hf_hub_download

_UNET_CACHE = {}  # lazy-loaded cache keyed by (filename_or_path, encoder_name)

def _load_unet_model(model_path=None, repo_id=None, filename=None, label="model", revision="main", force_download=False, encoder_name="vgg16"):
    """
    Load a binary Unet model from a local path or from Hugging Face Hub.
    Returns the model instance when successful, otherwise None.
    """
    cache_key = (model_path or filename, encoder_name)
    if cache_key in _UNET_CACHE:
        print(f"{label.capitalize()} served from cache.")
        return _UNET_CACHE[cache_key]

    model = Unet(encoder_name=encoder_name, encoder_weights="imagenet", in_channels=3, classes=1)
    resolved_path = None

    if model_path and os.path.exists(model_path):
        resolved_path = model_path
    elif repo_id and filename:
        try:
            resolved_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                force_download=force_download,
            )
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
        _UNET_CACHE[cache_key] = model
        return model
    except Exception as exc:
        print(f"Failed to load {label} from {resolved_path}: {exc}")
        return None


def segmentation_pipeline(
    folder_path=None,
    target_size=(256, 256),
    file_list=None,
    include_eyes=False,
    body_repo_id="markdanielarndt/Zebrafish_Segmentation",
    body_model_filename="best_model_body_3400_vgg19.pth",
    body_encoder_name="vgg19",
    body_revision="main",
    body_force_download=False,
    eye_model_path=None,
    eye_repo_id="markdanielarndt/Zebrafish_Segmentation",
    eye_model_filename="best_model_eye_3400.pth",
    eye_encoder_name="vgg16",
    include_edema=False,
    edema_model_path=None,
    edema_repo_id="markdanielarndt/Zebrafish_Segmentation",
    edema_model_filename="best_model_edema_3400_focal.pth",
    edema_encoder_name="vgg19",
):
    """
    Perform body segmentation on all images in the specified folder or file list.

    Pass either `folder_path` (directory) or `file_list` (sorted list of absolute paths).
    When `file_list` is provided it takes precedence and preserves the given order.

    Optional eye segmentation can be enabled by setting include_eyes=True.
    Optional edema segmentation can be enabled by setting include_edema=True.

    Returns:
        - default: (original_images, segmented_images, grown_images)
        - if include_eyes=True: (original_images, segmented_images, grown_images, eyes_images)
        - if include_eyes=True and include_edema=True: (original_images, segmented_images, grown_images, eyes_images, edema_images)
    """
    if file_list is not None:
        images = []
        for fp in file_list:
            img = cv2.imread(fp)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: could not load {fp}")
    else:
        images = load_images_from_path(folder_path)
    segmented_images = []
    grown_images = []
    original_images = []
    eyes_images = []
    edema_images = []

    print(f"Loading body segmentation model from {body_repo_id}/{body_model_filename} (revision={body_revision}, force_download={body_force_download})...")
    loaded_model = _load_unet_model(
        repo_id=body_repo_id,
        filename=body_model_filename,
        label="body model",
        revision=body_revision,
        force_download=body_force_download,
        encoder_name=body_encoder_name,
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
            encoder_name=eye_encoder_name,
        )
        if eyes_model is None:
            print(f"WARNING: Eye model unavailable at {eye_repo_id}/{eye_model_filename}. Returning empty eye masks.")
        else:
            print("Eye model loaded successfully!")

    edema_model = None
    if include_edema:
        print(f"Loading edema segmentation model from {edema_repo_id}/{edema_model_filename}...")
        edema_model = _load_unet_model(
            model_path=edema_model_path,
            repo_id=edema_repo_id,
            filename=edema_model_filename,
            label="edema model",
            encoder_name=edema_encoder_name,
        )
        if edema_model is None:
            print(f"WARNING: Edema model unavailable at {edema_repo_id}/{edema_model_filename}. Returning empty edema masks.")
        else:
            print("Edema model loaded successfully!")

    # Preprocessing parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    cv2_size = (target_size[1], target_size[0])  # cv2 uses (width, height)
    for img in images:
        original_image = np.array(img)
        img = cv2.resize(img, cv2_size, interpolation=cv2.INTER_LINEAR)

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

        if include_edema:
            if edema_model is not None:
                segmented_edema, _ = segment_fish(input_image, edema_model, biggest_only=False)
                segmented_edema_array = np.array(segmented_edema)
            else:
                segmented_edema_array = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)
            edema_images.append(segmented_edema_array)

        grown_images.append(grown_image)
        segmented_images.append(filled_image)
        original_images.append(original_image)

    if include_eyes and include_edema:
        return original_images, segmented_images, grown_images, eyes_images, edema_images

    if include_eyes:
        return original_images, segmented_images, grown_images, eyes_images

    return original_images, segmented_images, grown_images
