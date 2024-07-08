# Ball Tracking and Digit Recognition System

This project uses YOLO for ball detection and a pre-trained CNN model for digit recognition to track balls in a video and identify their movements between quadrants. Events such as ball entry and exit from quadrants are logged and saved.

Note: The processed video is not available here because upload limit by github you can access video [here](https://drive.google.com/drive/folders/1NKe6PAmLL2pnTDcZtgiYBZqghgCKMQpF?usp=sharing). 

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Code Structure](#code-structure)
- [Files and Directories](#files-and-directories)
- [Output](#output)
- [Acknowledgements](#acknowledgements)

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Pandas
- TensorFlow
- Keras
- Ultralytics YOLO

## Setup

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/ball-tracking-digit-recognition.git
    cd ball-tracking-digit-recognition
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the following files in place:
    - A trained YOLO model saved as `model/best.pt`
    - A digit recognition CNN model saved as `model/digit_recognition_model.h5`
    - A video file to process, named `video.mp4`

## How to Run

To start the ball tracking and digit recognition process, run the following command:

```bash
python main.py
```

## Code Structure

### main.py

This is the main script that performs the ball tracking and digit recognition.

- **Ball Class and Color Mapping**: Defines the classes and color mappings for the balls.
- **YOLO Model Loading**: Loads the pre-trained YOLO model for ball detection.
- **Video Capture**: Opens the video file for processing.
- **Digit Recognition Model Loading**: Loads the pre-trained CNN model for digit recognition.
- **detect_quadrant**: Function to detect quadrants using the digit recognition model.
- **draw_ball**: Function to draw bounding boxes and labels around detected balls.
- **track_balls**: Main function that processes the video frame by frame, detects balls, identifies quadrants, logs events, and saves the processed video.
- **save_event_data**: Function to save the logged events to a CSV file.

## Files and Directories

- **model/**
  - `best.pt`: Pre-trained YOLO model for ball detection.
  - `digit_recognition_model.h5`: Pre-trained CNN model for digit recognition.
- **video.mp4**: The video file to be processed.
- **output/**
  - `processed_video.mp4`: The video with tracked balls and quadrant information.
  - `event_data.csv`: CSV file containing the logged events.

## Output

The script generates two main outputs:

1. **Processed Video**: The video with bounding boxes and labels around detected balls, saved as `output/processed_video.mp4`.
2. **Event Data CSV**: A CSV file containing timestamps and details of ball entry and exit events, saved as `output/event_data.csv`.

The CSV file includes the following columns:

- **Time**: The timestamp of the event in seconds.
- **Quadrant**: The quadrant number where the event occurred.
- **Ball Color**: The color of the ball involved in the event.
- **Type**: The type of event (`Entry` or `Exit`).
