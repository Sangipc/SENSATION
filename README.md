# Exploring Drift Detection Methods for Blind Pedestrian Safety on Sidewalks

## Abstract
The primary objective of the project was to enhance the safety and independence of visually impaired individuals by developing a system capable of real-time detection and mitigation of "drift" during sidewalk navigation. Drift refers to the deviation from the intended path, which can pose significant challenges to visually impaired pedestrians. 

## Index Terms

* Depth Estimation
* Optical Flow
* Drift Detection
* HPC
* Computer Vision
* Visual Impairment Assistance

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Requirements](#requirements)
5. [Usage](#usage)
6. [Depth Estimation](#depth-estimation)
7. [Optical Flow](#optical-flow)
8. [Code Structure](#code-structure)

### Introduction

The project aims to provide a real-time assistive solution for visually impaired individuals by detecting drift from the intended walking path. The system employs depth estimation to assess the surrounding environment and optical flow to analyze movement patterns, offering guidance and alerts to users. 

#### Depth Estimation Models
The project explored several depth estimation models to detect drift. The main models included:

* **MiDaS**: A versatile model that provides relative depth information using a fully convolutional neural network architecture. MiDaS is particularly notable for its multi-scale approach, which is crucial for various applications, including navigation and 3D modeling​.
* **FastDepth**: Known for its efficiency and real-time capabilities, making it suitable for mobile applications.
* **SGDepth**: Offers robust depth estimation, suitable for detailed scene analysis.

MiDaS was identified as the most effective model due to its superior ability to detect subtle shifts in movement patterns, thereby accurately identifying instances of drift​.

#### Optical Flow Analysis

Along with depth estimation, the project utilized optical flow analysis to capture motion details between consecutive frames. Optical flow provides critical insights into the direction and magnitude of motion, which helps in detecting deviations from the expected path, also detects rotation movements with the specified angle of rotation. This method was essential for complementing depth estimation by offering a more comprehensive analysis of motion patterns.

* The project runs on High-Performance Computing (HPC) resources for efficient segmentation and real time analysis.

### Features

* Real-Time Depth Estimation: Utilizes the MiDaS model for accurate depth perception from video frames.
* Optical Flow Analysis: Tracks motion and detects drift to provide directional feedback.
* HPC Integration: Efficient processing of complex computations and large datasets.

### Installation

1. **Create a Virtual Environment**
   ```bash
   python -m venv env
2. **Activate the Virtual Environment**
  * On Windows:

    ```
    .\env\Scripts\activate
    ```
  * On macOS:

    ```
    source env/bin/activate
    ```
3. To set up the project, clone the repository

   ```
   git clone https://github.com/Sangipc/SENSATION.git
   ```

### Requirements

   ```
   pip install -r requirements.txt
   ```

### Usage

#### Depth Estimation

To run the depth estimation module:
  
  1. **Specify Input Video**
  
     Type the name of the input video file into input_file_name.txt.
  
  2. **Prepare Folders**
  
     Create the following directory structure inside Final_Folder:
    
      ```
      Final_Folder/
      ├── Input_Video/
      └── Output_Video/
      ```
  
  3. **Add Input Video**
  
      Place your input video into Final_Folder/Input_Video.
  
  4. **Run the Main Script**
  
      ```
      python main.py <path_to_Final_Folder> <path_to_DeeplabV3Plus_resnet50.pth>
      ```
  
      Replace <path_to_Final_Folder> with the path to your Final_Folder and <path_to_DeeplabV3Plus_resnet50.pth> with the path where the DeeplabV3Plus_resnet50.pth file is stored.
  
  5. **Check Output video** with highlighted depth maps which alerts the pedestrian of their walking.
  
      The processed output videos will be saved in Final_Folder/Output_Video.

#### Optical Flow

To run the optical flow analysis:
  
  ``` cd **Experiment_optical_flow** ```
  
  ```
  python main.py --input /path/to/video.mp4 --output /path/to/output_folder --model_path /path/to/segmentation_model --separate
  ```

* `--input`: Path to the input video file.
* `--output`: Directory for output files.
* `--model_path`: Path to the segmentation model.
* `--separate`: Optional flag to separate depth calculation for sidewalks.

### Code Structure

* `main.py`: Main script for starting the implementation. 
* `Test_on_videos.py`: Script for depth estimation and segmentation.
* `Experiment_optical_flow/main.py`: Main script for optical flow analysis and drift detection.
* `requirements.txt`: List of dependencies.
* `models/`: Directory containing pre-trained model weights for segmentation. I have used a pretrained DeepLabV3 model.
* `output/`: Directory for storing output frames and videos.
* `movements.txt`: Log of detected movements and angles.
