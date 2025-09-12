import numpy as np
import cv2
import os
from segmentation_models_pytorch import Unet
import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist
import torch.nn as nn
import timm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.transforms as T

def normalize_images(data):
        # Check if data contains np.arrays, if yes, directly normalize them
        if isinstance(data[0], np.ndarray):
            return np.array(data, dtype=np.float32)
        else:
            return np.array([np.array(image) for image in data], dtype=np.float32) 
        
def get_fish_length(img, original_size = [256,256], initial_size = (4090, 5420) ):
    image_array = np.array(img)

    # Calculate the maximum and minimum values of the image array
    max_value = np.max(image_array)
    min_value = np.min(image_array)
    
    # Calculate the image width and height
    image_height, image_width = image_array.shape
    # Resize the image to 256x256
    image_array = cv2.resize(image_array, (256, 256), interpolation=cv2.INTER_LINEAR)
    # Find the leftmost and rightmost white parts of the image
    white_rows = np.where(np.any(image_array == max_value, axis=1))[0]
    topmost_white = white_rows[0]
    bottommost_white = white_rows[-1]
    white_columns = np.where(np.any(image_array == max_value, axis=0))[0]
    leftmost_white = white_columns[0]
    rightmost_white = white_columns[-1]

    max_distance = 0
    for random_y1 in white_rows[::2]:
        for random_x1 in white_columns[::2]:
            if image_array[random_y1, random_x1] > 0:
                for random_y2 in white_rows[::2]:
                    for random_x2 in white_columns[::2]:
                        if image_array[random_y2, random_x2] > 0:
                            distance = np.hypot(
                                (random_x2 - random_x1) * (initial_size[0] / original_size[0]),
                                (random_y2 - random_y1) * (initial_size[1] / original_size[1])
                            )
                            if distance > max_distance:
                                max_distance = distance
                                x1, y1 = random_x1, random_y1
                                x2, y2 = random_x2, random_y2

    result_image = cv2.cvtColor(image_array.copy(), cv2.COLOR_GRAY2BGR)

    max_curve_length = np.inf
    found_valid_curve = False
    curve_type = "none"

    temp_curve_x = np.linspace(x1, x2, 100, dtype=int)
    temp_curve_y = np.linspace(y1, y2, 100, dtype=int)

    if image_array[temp_curve_y, temp_curve_x].all() > 0:
        curve_length = np.hypot(
            np.diff(temp_curve_x) * (initial_size[0] / original_size[0]),
            np.diff(temp_curve_y) * (initial_size[1] / original_size[1])
        ).sum()

        if curve_length < max_curve_length:
            max_curve_length = curve_length
            curve_x = temp_curve_x
            curve_y = temp_curve_y
            mid_point1 = None
            mid_point2 = None
            found_valid_curve = True
            curve_type = "no_midpoint"

    if not found_valid_curve:
        mid_y1, mid_x1 = np.meshgrid(
            np.arange(topmost_white, bottommost_white, 4),
            np.arange(leftmost_white, rightmost_white, 4),
            indexing='ij'
        )
        mid_y2, mid_x2 = np.meshgrid(
            np.arange(topmost_white, bottommost_white, 4),
            np.arange(leftmost_white, rightmost_white, 4),
            indexing='ij'
        )

        mid_y1 = mid_y1.ravel()
        mid_x1 = mid_x1.ravel()
        mid_y2 = mid_y2.ravel()
        mid_x2 = mid_x2.ravel()


        for i in range(len(mid_y1)):
            if image_array[mid_y1[i], mid_x1[i]] > 0:
                for j in range(len(mid_y2)):
                    if image_array[mid_y2[j], mid_x2[j]] > 0:
                        temp_curve_x1 = np.linspace(x1, mid_x1[i], 50, dtype=int)
                        temp_curve_y1 = np.linspace(y1, mid_y1[i], 50, dtype=int)
                        temp_curve_x2 = np.linspace(mid_x1[i], mid_x2[j], 50, dtype=int)
                        temp_curve_y2 = np.linspace(mid_y1[i], mid_y2[j], 50, dtype=int)
                        temp_curve_x3 = np.linspace(mid_x2[j], x2, 50, dtype=int)
                        temp_curve_y3 = np.linspace(mid_y2[j], y2, 50, dtype=int)

                        if (
                            image_array[temp_curve_y1, temp_curve_x1].all() > 0 and
                            image_array[temp_curve_y2, temp_curve_x2].all() > 0 and
                            image_array[temp_curve_y3, temp_curve_x3].all() > 0
                        ):
                            curve1_distance = np.hypot(
                                np.diff(temp_curve_x1) * (initial_size[0] / original_size[0]),
                                np.diff(temp_curve_y1) * (initial_size[1] / original_size[1])
                            ).sum()
                            curve2_distance = np.hypot(
                                np.diff(temp_curve_x2) * (initial_size[0] / original_size[0]),
                                np.diff(temp_curve_y2) * (initial_size[1] / original_size[1])
                            ).sum()
                            curve3_distance = np.hypot(
                                np.diff(temp_curve_x3) * (initial_size[0] / original_size[0]),
                                np.diff(temp_curve_y3) * (initial_size[1] / original_size[1])
                            ).sum()
                            curve_length = curve1_distance + curve2_distance + curve3_distance

                            if curve_length < max_curve_length:
                                max_curve_length = curve_length
                                curve_x1 = temp_curve_x1
                                curve_y1 = temp_curve_y1
                                curve_x2 = temp_curve_x2
                                curve_y2 = temp_curve_y2
                                curve_x3 = temp_curve_x3
                                curve_y3 = temp_curve_y3
                                mid_point1 = (mid_x1[i], mid_y1[i])
                                mid_point2 = (mid_x2[j], mid_y2[j])
                                found_valid_curve = True
                                curve_type = "two_midpoints"

    if curve_type == "no_midpoint":
        for i in range(len(curve_x) - 1):
            cv2.line(result_image, (curve_x[i], curve_y[i]), (curve_x[i + 1], curve_y[i + 1]), (255, 0, 0), thickness=1)

    elif curve_type == "two_midpoints":
        for i in range(len(curve_x1) - 1):
            cv2.line(result_image, (curve_x1[i], curve_y1[i]), (curve_x1[i + 1], curve_y1[i + 1]), (255, 0, 0), thickness=1)
        for i in range(len(curve_x2) - 1):
            cv2.line(result_image, (curve_x2[i], curve_y2[i]), (curve_x2[i + 1], curve_y2[i + 1]), (0, 255, 0), thickness=1)
        for i in range(len(curve_x3) - 1):
            cv2.line(result_image, (curve_x3[i], curve_y3[i]), (curve_x3[i + 1], curve_y3[i + 1]), (0, 0, 255), thickness=1)

    return result_image, max_curve_length

#Y_SCALE = 3923/256
#X_SCALE = 3923/256

Y_SCALE = 5420/256
X_SCALE = 4090/256


def get_fish_length_circles_fixed(body_mask, circle_dia=15):
    body_mask = np.array(body_mask)
    body_mask_bin = body_mask > 0
    eroded = binary_erosion(body_mask_bin)
    border_mask = body_mask_bin ^ eroded
    body_border = np.column_stack(np.where(border_mask))

    if len(body_border) == 0:
        return 0.0

    # start/end: farthest pair on the border
    if len(body_border) > 1:
        dists = cdist(body_border, body_border)
        max_idx = np.unravel_index(np.argmax(dists), dists.shape)
        start_point = body_border[max_idx[0]]
        farthest_border_point = body_border[max_idx[1]]
    else:
        start_point = farthest_border_point = body_border[0]

    image_height, image_width = body_mask.shape
    max_value = np.max(body_mask)

    circle_points = [tuple(start_point)]
    current_point = start_point.copy()

    while True:
        dist_to_end = np.linalg.norm(current_point - farthest_border_point)

        # If we’re within one step, append the end point and finish
        if dist_to_end <= circle_dia:
            circle_points.append(tuple(farthest_border_point))
            break

        theta = np.linspace(0, 2*np.pi, 360)
        circle_x = (current_point[1] + circle_dia * np.cos(theta)).astype(int)
        circle_y = (current_point[0] + circle_dia * np.sin(theta)).astype(int)
        valid_idx = (circle_x >= 0) & (circle_x < image_width) & (circle_y >= 0) & (circle_y < image_height)
        circle_x = circle_x[valid_idx]
        circle_y = circle_y[valid_idx]

        # stay on mask border (or at least on mask)
        mask_points = body_mask[circle_y, circle_x] == max_value
        border_circle_x = circle_x[mask_points]
        border_circle_y = circle_y[mask_points]

        if len(border_circle_x) == 0:
            # No candidates; snap to end and finish
            circle_points.append(tuple(farthest_border_point))
            break

        candidates = np.column_stack((border_circle_y, border_circle_x))

        if len(circle_points) > 1:
            prev_point = np.array(circle_points[-2])
            d_prev = np.linalg.norm(candidates - prev_point, axis=1)
            d_curr = np.linalg.norm(candidates - current_point, axis=1)
            forward_mask = d_curr < d_prev
            candidates = candidates[forward_mask]
            if len(candidates) == 0:
                # If forward filter removes all, pick the one most aligned towards the end
                candidates = np.column_stack((border_circle_y, border_circle_x))

        # Pick the candidate most aligned with the vector to the end (greedy towards end)
        to_end_vec = (farthest_border_point - current_point).astype(float)
        to_end_vec /= (np.linalg.norm(to_end_vec) + 1e-9)
        cand_vecs = (candidates - current_point).astype(float)
        cand_vecs /= (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-9)
        best_idx = np.argmax((cand_vecs @ to_end_vec))
        next_point = candidates[best_idx]

        circle_points.append(tuple(next_point))
        current_point = next_point

    # Compute physical length with correct axes
    circle_points_arr = np.array(circle_points, dtype=float)
    diffs = np.diff(circle_points_arr, axis=0)
    # diffs[:,0] is row (y), diffs[:,1] is col (x)
    diffs[:, 0] *= Y_SCALE
    diffs[:, 1] *= X_SCALE
    segment_lengths = np.linalg.norm(diffs, axis=1)
    path_length = float(np.sum(segment_lengths))
    return path_length, np.array(circle_points)

def get_fish_length_circles(body_mask):
    body_mask = np.array(body_mask)

    body_mask_bin = body_mask > 0
    eroded = binary_erosion(body_mask_bin)
    border_mask = body_mask_bin ^ eroded
    body_border = np.column_stack(np.where(border_mask))
    # Find the two border points that are farthest apart
    if len(body_border) > 1:
        dists = cdist(body_border, body_border)
        max_idx = np.unravel_index(np.argmax(dists), dists.shape)
        start_point = body_border[max_idx[0]]
        farthest_border_point = body_border[max_idx[1]]
    else:
        start_point = farthest_border_point = body_border[0]


    distances_from_start = np.linalg.norm(body_border - start_point, axis=1)
    max_idx = np.argmax(distances_from_start)
    farthest_border_point = body_border[max_idx]
    image_array = body_mask
    max_value = np.max(image_array)

    image_height, image_width = image_array.shape
    circle_points = []
    current_point = start_point.copy()
    circle_points.append(tuple(current_point))
    circle_dia = 40
    while True:
        dist_to_end = np.linalg.norm(current_point - farthest_border_point)
        if dist_to_end < circle_dia:
            break
        theta = np.linspace(0, 2 * np.pi, 360)
        circle_x = (current_point[1] + circle_dia * np.cos(theta)).astype(int)
        circle_y = (current_point[0] + circle_dia * np.sin(theta)).astype(int)
        valid_idx = (circle_x >= 0) & (circle_x < image_width) & (circle_y >= 0) & (circle_y < image_height)
        circle_x = circle_x[valid_idx]
        circle_y = circle_y[valid_idx]
        mask_points = image_array[circle_y, circle_x] == max_value
        border_circle_x = circle_x[mask_points]
        border_circle_y = circle_y[mask_points]
        if len(border_circle_x) == 0:
            break
        if len(circle_points) > 1:
            prev_point = np.array(circle_points[-2])
            candidates = np.column_stack((border_circle_y, border_circle_x))
            dists_to_prev = np.linalg.norm(candidates - prev_point, axis=1)
            dists_to_curr = np.linalg.norm(candidates - current_point, axis=1)
            forward_mask = dists_to_curr < dists_to_prev
            candidates = candidates[forward_mask]
            if len(candidates) == 0:
                break
        else:
            candidates = np.column_stack((border_circle_y, border_circle_x))
        mid_idx = len(candidates) // 2
        next_point = candidates[mid_idx]
        circle_points.append(tuple(next_point))
        current_point = next_point
    circle_points_arr = np.array(circle_points)
    scaled_diffs = np.diff(circle_points_arr, axis=0).astype(float)
    scaled_diffs[:, 0] *= 4090/256  # x direction (column)
    scaled_diffs[:, 1] *= 5420/256  # y direction (row)
    segment_lengths = np.linalg.norm(scaled_diffs, axis=1)
    path_length = np.sum(segment_lengths)
    return path_length

def apply_mask(original_image, mask):
    """
    Apply the mask to the original image.
    """
    # Convert the mask to a 3-channel image
    original_image = cv2.resize(original_image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Invert the mask so that the fish is white (255) and background is black (0)
    #mask = cv2.bitwise_not(mask)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(original_image, mask_3ch)

    return masked_image

def apply_mask2(original_image, mask):
    """
    Apply the mask to the original image, setting background pixels to transparent (if supported) or a specific color.
    """
    # Resize original image to match mask
    original_image = cv2.resize(original_image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

    # If you want transparency, create a 4-channel image (BGRA)
    if len(original_image.shape) == 2 or original_image.shape[2] == 3:
        # Convert to BGRA
        masked_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)
    else:
        masked_image = original_image.copy()

    # Create alpha channel from mask: foreground=255, background=0
    alpha = np.where(mask > 0, 255, 0).astype(np.uint8)
    masked_image[:, :, 3] = alpha

    # If you want a specific background color instead of transparency, uncomment below:
    # bg_color = [255, 255, 255]  # white background
    # for c in range(3):
    #     masked_image[:, :, c][mask == 0] = bg_color[c]
    # masked_image = masked_image[:, :, :3]  # remove alpha if not needed


    return masked_image

def classification_curvature(image, mask, model, use_threshold, threshold):
    
    masked_image = apply_mask(image, mask)

    cropped_image = preprocess_masked_image(masked_image)

    # Ensure the masked image is in RGB format
    masked_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    
    # Ensure the image is scaled to [0, 255] before preprocessing
    masked_image_rgb = np.clip(masked_image_rgb, 0, 255).astype(np.uint8)
    
    # Preprocess the image
    processed_image = normalize_images([masked_image_rgb])

    processed_image = T.ToPILImage()(processed_image[0])
    processed_image = T.ToTensor()(processed_image)
    processed_image = processed_image.unsqueeze(0)
    #processed_image = torch.from_numpy(processed_image).permute(0, 3, 1, 2).float()
    processed_image = processed_image.to(device)
    
    outputs = model(processed_image)
    curvature = 1 + torch.argmax(outputs, dim=1)

    probs = F.softmax(outputs, dim=1)
    confs, preds = torch.max(probs, dim=1)

    if use_threshold:
        if confs < threshold:
            curvature = torch.tensor([5]).to(device)

    return cropped_image, curvature

class FishClassifier(nn.Module):
    def __init__(self, num_classes, dense_layer_size, dropout_rate, model_name='resnet101'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.flatten = nn.Flatten()
        # Get backbone output feature size by passing a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)
            dummy_output = self.backbone(dummy_input)
            backbone_out_features = dummy_output.shape[1] if len(dummy_output.shape) > 1 else dummy_output.shape[0]
        self.fc1 = nn.Linear(backbone_out_features, dense_layer_size)
        #self.fc1 = nn.Linear(self.backbone.num_features, dense_layer_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dense_layer_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model(cnn_log_directory="Models/CNN", cnn_model_name="ResNet50x1/0006.keras"):
    model_path = os.path.join(cnn_log_directory, cnn_model_name)
    # Path to saved model
    model_path = "runs/f1_based_dropout/best_model_class.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found in working directory.")


    fallback = {'dense_layer': 512, 'dropout': 0.2, 'model_name': 'convnext_base'}
    print("Warning: best_params not found. Using fallback params:", fallback)
    best_params = fallback

    # Instantiate and load
    model = FishClassifier(num_classes=4,
                        dense_layer_size=best_params['dense_layer'],
                        dropout_rate=best_params['dropout'],
                        model_name=best_params['model_name'])
    try:
        # # Try loading state dict first
        state = torch.load(model_path, map_location=device)
        #if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        #     model.load_state_dict(state)
        # else:
        #     # If saved the entire model object
        model = state
    except Exception as e:
        # Last resort: try direct load_state_dict on the object
        model.load_state_dict(torch.load(model_path, map_location=device))
    # model = model.to(device)
    # Ensure `model` is an nn.Module on the correct device and in eval mode.

    def _strip_module_prefix(state_dict):
        if any(k.startswith('module.') for k in state_dict.keys()):
            return {k.replace('module.', ''): v for k, v in state_dict.items()}
        return state_dict

    if isinstance(model, nn.Module):
        model = model.to(device)
        model.eval()
    else:
        # model is likely a state_dict / OrderedDict -> instantiate and load
        state_dict = model
        if isinstance(state_dict, (dict, OrderedDict)):
            state_dict = _strip_module_prefix(state_dict)
            instantiated = FishClassifier(
                num_classes=4,
                dense_layer_size=best_params['dense_layer'],
                dropout_rate=best_params['dropout'],
                model_name=best_params['model_name']
            )
            # handle common nested keys
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                state_dict = _strip_module_prefix(state_dict)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
                state_dict = _strip_module_prefix(state_dict)
            instantiated.load_state_dict(state_dict)
            model = instantiated.to(device)
            model.eval()
        else:
            raise TypeError("Loaded object is neither an nn.Module nor a state dict.")

    return model


def plot_edges_with_curvature(mask, min_contour_length, window_size_ratio):
    # Compute edge properties
    edge_pixels, curvature_values = compute_curvature_profile(mask, min_contour_length, window_size_ratio)

    # Plot the mask
    plt.imshow(mask, cmap='gray')
    # We set the min and max of the colorbar, so that 90% of the curvature values are shown.
    # This is to have a nice visualization. You can change this threshold according to your specific task.
    threshold = np.percentile(np.abs(curvature_values), 90)
    plt.scatter(edge_pixels[:, 1], edge_pixels[:, 0], c=curvature_values, cmap='jet', s=5, vmin=-threshold, vmax=threshold)

    plt.colorbar(label='Curvature')
    plt.title("Curvature of Edge Pixels")
    plt.show()
    return curvature_values

def compute_curvature_profile(mask, min_contour_length, window_size_ratio):
    # Compute the contours of the mask to be able to analyze each part individually
    contours = measure.find_contours(mask, 0.5)

    # Initialize arrays to store the curvature information for each edge pixel
    curvature_values = []
    edge_pixels = []

    # Iterate over each contour
    for contour in contours:
        # Iterate over each point in the contour
        for i, point in enumerate(contour):
            # We set the minimum contour length to 20
            # You can change this minimum-value according to your specific requirements
            if contour.shape[0] > min_contour_length:
                # Compute the curvature for the point
                # We set the window size to 1/5 of the whole contour edge. Adjust this value according to your specific task
                window_size = int(contour.shape[0]/window_size_ratio)
                curvature = compute_curvature(point, i, contour, window_size)
                # We compute, whether a point is convex or concave.
                # If you want to have the 2nd derivative shown you can comment this part
                # if curvature > 0:
                #     curvature = 1
                # if curvature <= 0:
                #     curvature = -1
                # Store curvature information and corresponding edge pixel
                curvature_values.append(curvature)
                edge_pixels.append(point)

    # Convert lists to numpy arrays for further processing
    curvature_values = np.array(curvature_values)
    edge_pixels = np.array(edge_pixels)

    return edge_pixels, curvature_values


def compute_curvature(point, i, contour, window_size):
    # Compute the curvature using polynomial fitting in a local coordinate system

    # Extract neighboring edge points
    start = max(0, i - window_size // 2)
    end = min(len(contour), i + window_size // 2 + 1)
    neighborhood = contour[start:end]

    # Extract x and y coordinates from the neighborhood
    x_neighborhood = neighborhood[:, 1]
    y_neighborhood = neighborhood[:, 0]

    # Compute the tangent direction over the entire neighborhood and rotate the points
    tangent_direction_original = np.arctan2(np.gradient(y_neighborhood), np.gradient(x_neighborhood))
    tangent_direction_original.fill(tangent_direction_original[len(tangent_direction_original)//2])

    # Translate the neighborhood points to the central point
    translated_x = x_neighborhood - point[1]
    translated_y = y_neighborhood - point[0]


    # Apply rotation to the translated neighborhood points
    # We have to rotate the points to be able to compute the curvature independent of the local orientation of the curve
    rotated_x = translated_x * np.cos(-tangent_direction_original) - translated_y * np.sin(-tangent_direction_original)
    rotated_y = translated_x * np.sin(-tangent_direction_original) + translated_y * np.cos(-tangent_direction_original)

    # Fit a polynomial of degree 2 to the rotated coordinates
    coeffs = np.polyfit(rotated_x, rotated_y, 2)


    # You can compute the curvature using the formula: curvature = |d2y/dx2| / (1 + (dy/dx)^2)^(3/2)
    # dy_dx = np.polyval(np.polyder(coeffs), rotated_x)
    # d2y_dx2 = np.polyval(np.polyder(coeffs, 2), rotated_x)
    # curvature = np.abs(d2y_dx2) / np.power(1 + np.power(dy_dx, 2), 1.5)
    # We compute the 2nd derivative in order to determine whether the curve at the certain point is convex or concave
    curvature = np.polyval(np.polyder(coeffs, 2), rotated_x)

    # Return the mean curvature for the central point
    return np.mean(curvature)

# Set minimum length of the contours that should be analyzed
min_contour_length = 20
# Set the ratio of the window size (contour length / window_size_ratio) for local polynomial approximation
window_size_ratio = 5

def preprocess_masked_image(image, target_size=(256, 256)):
    """
    Preprocess a single masked image by cropping to the bounding box, 
    padding to a square, and resizing to the target size.

    Args:
        image (numpy array): The input masked image.
        target_size (tuple): The desired output size (width, height).

    Returns:
        numpy array: The processed image.
    """
    # Step 1: Convert to grayscale and find non-black pixels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)

    if coords is None:  # Image is fully black
        print("The image is fully black and cannot be processed.")
        return None

    # Step 2: Crop to bounding box
    x, y, w, h = cv2.boundingRect(coords)
    cropped_image = image[y:y+h, x:x+w]

    # Step 3: Pad to square size
    height, width = cropped_image.shape[:2]
    max_dim = max(height, width)
    pad_top = (max_dim - height) // 2
    pad_bottom = max_dim - height - pad_top
    pad_left = (max_dim - width) // 2
    pad_right = max_dim - width - pad_left

    padded_image = cv2.copyMakeBorder(
        cropped_image, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Step 4: Resize
    resized_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_LINEAR)

    return resized_image

def preprocess_masked_image2(image, target_size=(256, 256)):
    """
    Preprocess a single masked image with transparency (BGRA), cropping to the bounding box of non-transparent pixels,
    padding to a square, and resizing to the target size.

    Args:
        image (numpy array): The input masked image (BGRA).
        target_size (tuple): The desired output size (width, height).

    Returns:
        numpy array: The processed image.
    """
    # Step 1: Find non-transparent pixels (alpha > 0)
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        coords = cv2.findNonZero((alpha > 0).astype(np.uint8))
    else:
        # Fallback to grayscale for non-BGRA images
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)

    if coords is None:  # Image is fully transparent/black
        print("The image is fully transparent/black and cannot be processed.")
        return None

    # Step 2: Crop to bounding box
    x, y, w, h = cv2.boundingRect(coords)
    cropped_image = image[y:y+h, x:x+w]

    # Step 3: Pad to square size
    height, width = cropped_image.shape[:2]
    max_dim = max(height, width)
    pad_top = (max_dim - height) // 2
    pad_bottom = max_dim - height - pad_top
    pad_left = (max_dim - width) // 2
    pad_right = max_dim - width - pad_left

    padded_image = cv2.copyMakeBorder(
        cropped_image, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0, 0] if cropped_image.shape[2] == 4 else [0, 0, 0]
    )

    # Step 4: Resize
    resized_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_LINEAR)

    return resized_image