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

## Table of Contents
- [How to Use](#how-to-use)
  - [Uploading Images](#uploading-images)
    - [Method 1: Upload a Folder (Preferred)](#method-1-upload-a-folder-preferred)
    - [Method 2: Upload Individual Images](#method-2-upload-individual-images)
  - [Selecting Endpoints](#selecting-endpoints)
  - [Results and Downloads](#results-and-downloads)

## How to Use

### Uploading Images

You can upload images of the zebrafish in two ways:

#### Method 1: Upload a Folder (Preferred)

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

### Selecting Endpoints

After uploading your images, choose which endpoints you want to analyze:

![Select endpoints](Documentation_images/screenshot8.png)

You can select:
- **Length**: Measure the length of the zebrafish
- **Curvature**: Classify the zebrafish into curvature classes (1-4)
  - Class 1: Most severe curvature
  - Class 2: Moderate-severe curvature
  - Class 3: Mild curvature
  - Class 4: Most healthy (minimal curvature)

You can also choose whether to enable **Threshold/Human-in-the-Loop mode** and set a threshold value. This mode allows for manual review of uncertain predictions (more details below).

**Threshold / Human-in-the-Loop (Curvature only)**

If you activate threshold mode you may set a confidence threshold between 0.0 and 1.0. The curvature classifier provides an inherent confidence score for its predicted curvature label; if the model's confidence for an image does not exceed the chosen threshold, that image will not be assigned a curvature class and will instead be reported as "Not Classified" in the Excel output. This thresholding applies only to curvature classification, length measurements are not affected.

Example: see ![Documentation_images/screenshot12.png](Documentation_images/screenshot12.png). The left side shows the Excel output with threshold mode disabled (all images receive a curvature class). The right side shows threshold mode enabled with a threshold of 0.95: several images have confidence below 0.95 and are therefore marked "Not Classified", allowing those cases to be routed for manual review.

Use a higher threshold to reduce automatic curvature assignments and increase human review of uncertain cases; choose a lower threshold to classify more images automatically.

### Results and Downloads

After processing, you can download an Excel sheet containing individual fish annotations:

![Download Excel](Documentation_images/screenshot9.png)

Below the download button, you'll see boxplots visualizing the distribution of the selected endpoints. These boxplots are also included in the Excel file:

![Boxplots](Documentation_images/screenshot10.png)

#### Segmentation Preview

A preview of the first 5 segmentations is displayed below the boxplots:

![Segmentation preview](Documentation_images/screenshot11.png)

**Important:** Verify that the segmentations accurately match the fish. If the segmentations are incorrect or misaligned, the resulting endpoint measurements will be inaccurate. If you notice issues with the segmentations, you may need to adjust your images or use the threshold mode for manual review.

---
