# 🌍 EnviroWatch - HackOut'24

**Explore the future of NDVI Change Detection with advanced predictive analytics for agricultural health and land-use changes.**

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Usage](#usage)
5. [Data Pipeline](#data-pipeline)
6. [Results](#results)
7. [Future Enhancements](#future-enhancements)


## Project Overview

**EnviroWatch - HackOut'24** is an innovative platform developed for monitoring and analyzing agricultural health and land-use changes over time. By leveraging satellite imagery, EnviroWatch aims to detect, visualize, and predict changes in vegetation. This tool assists in understanding the impacts of agricultural practices, climate change, and deforestation, thereby enabling more informed and proactive decision-making for sustainable agriculture.

### Problem Statement

Agricultural health and land-use changes are critical indicators of environmental sustainability. Traditional methods of monitoring vegetation changes are often time-consuming and lack the predictive capabilities necessary for proactive measures. EnviroWatch addresses this gap by providing a robust, data-driven solution for detecting and forecasting vegetation changes.

## Features

- **🛰️ Satellite Image Upload:** Upload satellite images from different years to detect changes in vegetation.
- **🔍 NDVI Calculation:** Automatically calculate NDVI (Normalized Difference Vegetation Index) to assess vegetation health.
- **📊 Predictive Analytics:** Implement machine learning models to predict future vegetation changes based on historical data.
- **🌱 Vegetation Change Visualization:** Visualize changes in vegetation over time, highlighting areas of degradation or improvement.
- **🤖 Interactive Chatbot:** Get guidance and insights about the platform's features through an AI-driven chatbot.
- **🗺️ GeoTIFF Analysis and Image Classification:** Extract detailed metadata from GeoTIFF images and analyze them using a generative AI model to provide actionable insights. This includes:
  - **Metadata Extraction:** Extracts CRS, transformation matrix, bounding box, dimensions, band information, and additional band statistics from the GeoTIFF file.
  - **Geographic Conversion:** Converts pixel coordinates to geographic coordinates (latitude and longitude).
  - **Location Lookup:** Retrieves location information using geographic coordinates.
  - **Image Analysis:** Analyzes the image with the extracted metadata using the gemini-1.5-flash AI model to provide intelligent suggestions for land transformation and deforestation prevention.

## Tech Stack

- **Frontend:**
  - HTML, TailwindCSS, JavaScript
- **Backend:**
  - Flask
  - GDAL/GeoPandas for geospatial data processing
- **DevOps:**
  - Docker for containerization

## Usage

### Uploading Images

1. Navigate to the **Predict Changes** section.
2. Upload satellite images from different years using the provided form.
3. Click **Submit** to initiate the analysis.

### Predictive Analytics

1. Go to the **Predict Changes** section.
2. The platform will calculate NDVI and compare the images to detect vegetation changes.
3. View the results in a detailed visual report, highlighting areas of significant change.

### GeoTIFF Analysis and Image Classification

1. Navigate to the **GeoTIFF Analysis** section.
2. Upload a GeoTIFF image file.
3. The platform will automatically extract metadata, convert geographic coordinates, and perform a detailed analysis using the gemini-1.5-flash AI model.
4. Review actionable insights for land transformation and deforestation prevention.

### Chatbot Assistance

- Access the chatbot from the **Chatbot** section to get step-by-step guidance on how to use the platform or understand the results.

## Data Pipeline

The data pipeline in EnviroWatch is designed to efficiently process and analyze satellite imagery and related datasets.

1. **Data Ingestion:**
   - Upload satellite images (TIFF or TIF).

2. **Data Processing:**
   - **Image Preprocessing:** Resampling, registration, and NDVI calculation.
   - **Feature Engineering:** Extract temporal and spatial features and metadata from NDVI and climate data.

3. **Visualization:**
   - Display change detection results and predictive analytics in a user-friendly dashboard.

## Results

- **Vegetation Change Detection:** Accurately identified regions with significant vegetation loss or gain.
- **Predictive Analytics:** Provided reliable forecasts of future vegetation conditions, enabling early intervention.
- **GeoTIFF Analysis:** Delivered detailed insights on land transformation and deforestation risks through advanced image classification.

## Future Enhancements

- **Real-time Monitoring:** Integrate real-time data streams for continuous monitoring and prediction.
- **Enhanced Visualization:** Develop 3D visualizations of vegetation changes for better interpretation.
- **User Customization:** Allow users to set custom thresholds for change detection and alerts.
- **Expanded Dataset:** Incorporate additional environmental variables like soil quality, air pollution, etc., for a more comprehensive analysis.
