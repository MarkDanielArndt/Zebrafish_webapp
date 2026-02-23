import os
from PIL import Image
import torch
import numpy as np
import cv2

def load_images_from_path(path):
    """
    Loads all images from the specified directory.

    Args:
        path (str): The directory path containing the images.

    Returns:
        list: A list of PIL Image objects.
    """
    images = []
    for file_name in os.listdir(path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', 'tif')):
            file_path = os.path.join(path, file_name)
            try:
                #img = Image.open(file_path)
                img = cv2.imread(file_path)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {file_name}: {e}")
    return images

def segment_fish(image, model, biggest_only=True):
    """
    Segment fish from the image using a pre-trained Unet model.
    
    Parameters:
        image (PIL.Image): The input image.
        
    Returns:
        PIL.Image: The segmented image with fish highlighted.
    """
    # Transform the image
    #image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image
    
    with torch.no_grad():
        # Get predictions from the model
        prediction = model(image_tensor)
    
    # Get the mask
    mask = prediction.squeeze().cpu().numpy()
    # Convert the mask to a confidence map
    confidence_map = (mask - mask.min()) / (mask.max() - mask.min()) * 255
    confidence_map = confidence_map.astype(np.uint8)
    
    # Convert the confidence map to a binary mask
    binary_mask = (confidence_map > 127).astype(np.uint8) * 255
    
    # Convert the binary mask to a PIL image
    if biggest_only:
        # Find the largest connected component in the binary mask
        num_labels, labels_im = cv2.connectedComponents(binary_mask)

        # Find the largest component
        largest_component = 1 + np.argmax(np.bincount(labels_im.flat)[1:])

        # Create a mask for the largest component
        largest_component_mask = (labels_im == largest_component).astype(np.uint8) * 255

        # Convert the largest component mask to a PIL image
        segmented_image = Image.fromarray(largest_component_mask)
    else:
        # Keep all components
        segmented_image = Image.fromarray(binary_mask)
    
    return segmented_image, confidence_map


def fill_holes(mask):
    """
    Fill all holes in a binary mask (0-1) by flood filling from the background.
    Tries top-left corner first, falls back to bottom-right if needed.
    """
    mask = (mask > 0).astype(np.uint8)  # Ensure the mask is uint8 (0-1 range)
    h, w = mask.shape

    flood_filled = mask.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)

    # Determine a background seed point (top-left or bottom-right)
    if mask[0, 0] == 0:
        seed = (0, 0)
    elif mask[h - 1, w - 1] == 0:
        seed = (w - 1, h - 1)
    else:
        return mask  # Return the original mask if no safe corner found

    # Perform flood fill
    cv2.floodFill(flood_filled, mask_ff, seedPoint=seed, newVal=1)

    # Invert the flood fill result (this now represents the "holes")
    flood_filled_inv = 1-flood_filled
        
    # Combine the original mask with the holes to a new mask: result (0-1 range)
    filled_mask = mask | flood_filled_inv
    
    return (filled_mask * 255).astype(np.uint8)

def grow_mask(mask, iterations=3, kernel_size=3):
    """
    Dilate the mask to grow the region by a number of iterations.

    Args:
    - mask: binary mask (numpy array with 0 and 1 or 0 and 255)
    - iterations: how many times to apply dilation
    - kernel_size: size of the structuring element

    Returns:
    - grown mask (same shape)
    """
    # Ensure binary with values 0 and 1
    mask = (mask > 0).astype(np.uint8)

    # Create a structuring element (you can also try cv2.MORPH_RECT or MORPH_CROSS)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply dilation
    dilated = cv2.dilate(mask, kernel, iterations=iterations)

    return dilated*255


