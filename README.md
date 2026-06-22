---
title: Zebrafish Segmentation Web App
emoji: 🐟
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.3.0"
app_file: app.py
pinned: false
---

# 🐟 Zebrafish Segmentation Web App

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg.svg)](https://huggingface.co/spaces/markdanielarndt/Zebrafish)

## Table of Contents
- [How to use the Zebrafish Segmentation Web App](#How-to-use-the-Zebrafish-Segmentation-Web-App)
  - [Selecting a Model](#selecting-a-model)
  - [Uploading Images](#uploading-images)
    - [Method 1: Upload a Folder (Preferred)](#method-1-upload-a-folder-preferred)
    - [Method 2: Upload Individual Images](#method-2-upload-individual-images)
  - [Scale Bar Calibration](#scale-bar-calibration) 
  - [Selecting Endpoints](#selecting-endpoints)
  - [Run](#run)
  - [Boxplots](#boxplots)
  - [Segmentation Preview](#segmentation-preview)
  - [Generating a Final Excel](#generating-a-final-excel)

## How to use the Zebrafish Segmentation Web App

### Selecting a Model

Before you begin, you can select the model that will analyze the images. The General Model is trained on a broad dataset. Fine-tuned models are optimized for specific imaging setups. In addition, you can choose between a faster model and a more accurate model.

![Model selection option](Documentation_images/screenshot_model_selector_21_05.png)

### Uploading Images

You can upload images of the zebrafish in two ways:

#### Method 1: Upload a Folder

You can upload an entire folder containing zebrafish images.

![Upload folder option](Documentation_images/screenshot1.png)

Select the folder of your choosing from the file dialog:

![Select folder](Documentation_images/screenshot3.png)

Then click "Upload" to upload the folder. You'll need to confirm the upload by clicking "Upload" again:

![Confirm upload](Documentation_images/screenshot5.png)

Wait for the images to load and appear:

![Images loaded](Documentation_images/screenshot6.png)

#### Method 2: Upload Individual Images

Alternatively, you can upload individual images one by one.

![Upload individual images option](Documentation_images/screenshot2.png)

Select the images of your choosing and click "Open" to upload them:

![Select individual images](Documentation_images/screenshot4.png)

Wait for the images to load and appear:

![Images loaded](Documentation_images/screenshot7.png)

### Scale Bar Calibration

Once the images are successfully uploaded, the first one will appear on the screen. The scale bar will be automatically detected and the number of pixels it spans will be displayed. 

![Scale Bar Automatic Detection](Documentation_images/screenshot_auto_scale_detection.png)

Enter the physical length printed above the bar (e.g. 500) and its unit, then click Apply to compute the µm/px calibration. The image width/height fields below will be filled automatically.

![Length Values](Documentation_images/screenshot_length_values.png)

If automatic detection has incorrectly identified the scale bar, you can use the Manual Scale Bar Entry option below. The image loads automatically below. Click on it to mark the two endpoints of the scale bar line:
- 1st click → START (one end, shown in green)
- 2nd click → END (other end, shown in red)

![Manual Scale Bar Entry](Documentation_images/screenshot_manual_scalebar_entry.png)

After both endpoints are set, enter the physical length below and click Apply Manual Points to fill in the calibration automatically.

![Apply Manual Points](Documentation_images/screenshot_aply_manual_points.png)

You can also skip this step and enter the distances manually.

### Selecting Endpoints

After uploading your images, choose which endpoints you want to analyze:

![Select endpoints](Documentation_images/screenshot_endpoints_21_05.png)

You can select:
- **Length**: Measure the length of the zebrafish (centerline path length in µm)
- **Curvature**: Classify the zebrafish into curvature classes (1-4)
  - Class 1: Most severe curvature
  - Class 2: Moderate-severe curvature
  - Class 3: Mild curvature
  - Class 4: Most healthy (minimal curvature)
- **Length/Straight Line Ratio**: The ratio between the actual centerline length and the straight-line distance between endpoints. A value close to 1.0 indicates a nearly straight fish, while higher values indicate more curvature. This metric quantifies body curvature independently of fish size.
- **Eye Size**: Calculates the eye area in µm² and measures the eye diameter in µm. 
- **Edema**: Calculates the edema area in µm² .

### Run

To start the processing, click the **Run** button.

![Run](Documentation_images/screenshot_run.png)

### Boxplots

After processing, you'll see boxplots visualizing the distribution of the selected endpoints. These boxplots are also included in the Excel file:

![Boxplots](Documentation_images/screenshot_box_19_06.webp)

The boxplots display:
- **Fish Lengths**: Distribution of measured centerline lengths in µm
- **Curvatures**: Distribution of curvature classifications (1-4)
- **Length/Straight Line Ratio**: Distribution of the length ratio metric, where values closer to 1.0 indicate straighter fish
- **Eye Area**: Distribution of calculated eye areas in µm²
- **Edema Area**: Distribution of calculated edema areas in µm²

The ratio visualization helps identify fish with significant body curvature. You can see in the example image that some fish have ratios above 1.0, indicating curved body shapes:

![Length/Straight Line Ratio explanation](Documentation_images/screenshot_rel_length.png)

The cyan line shows the straight-line distance between endpoints, while the actual centerline path (shown in red) is longer due to body curvature. The ratio quantifies this difference.

### Segmentation Preview

A gallery displays segmentation overlays for the uploaded images (thumbnails include a short filename label). 

![Segmentation preview](Documentation_images/screenshot_segmentation_preview.png)

If automatic detection fails use Manual Point Adjustment tool to manually set head and tail points. 

- Click an image in the gallery above to select it for manual editing
- Click on the large image below to set HEAD (green) and TAIL (red) points
- Click 'Apply Manual Points' to recalculate the length

![Manual points](Documentation_images/screenshot_manual_point.png)

If you want to correct an incorrect segmentation by yourself, use the **Manual Mask Editor tool**.
After selecting the required image from the gallery at the top, choose the appropriate layer: body, eye, or edema, and click **Load Image into Editor**. 

![Mask Editor](Documentation_images/screenshot_mask_editor.png)

The tool selection is located on the left. To draw in missing fragments, select the brush tool and the yellow color. To remove excess parts, paint over them with blue. The eraser tool is used to erase the strokes you just drew. 

![Mask Editor Tools](Documentation_images/screenshot_mask_editor_tools.png)

You can also adjust the size of the selected tool by clicking the button at the bottom of the toolbar.

![Mask Editor Tools Size](Documentation_images/screenshot_mask_editor_tools_size.png)

To erase everything you’ve drawn, click the **Reset** button. 
To apply the changes, click **Apply**. After that, the canvas will be cleared, and the updated image will appear in the gallery at the top.


### Generating a Final Excel

First, select the desired image from the preview at the top, and then click the checkboxes next to the endpoints you want to exclude from the final statistics. Then press the **Save Exclusions for This Image** button (the row will still appear but the cell will say Excluded and it won't count toward statistics).

![Exclude Measurements for This Image](Documentation_images/screenshot_exclude_excel.png)

In the field below, you can enter sheet name that will be generated.

![Sheet name](Documentation_images/screenshot_sheet_name.png)

When you're ready, press the **Generate Final Excel** button. 

Below will be a button to download the file.

![Download Final Excel](Documentation_images/screenshot_download_excel.png)
