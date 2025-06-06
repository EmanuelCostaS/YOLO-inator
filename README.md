# Video to YOLO Annotation Conversion Tool

This project provides a set of Python scripts to convert video annotations from a specific CSV format into the YOLO (You Only Look Once) format required for object detection model training.  
It includes a script for processing a single video and a batch-processing script to handle multiple video sessions efficiently.

The scripts are designed to be robust, handling common issues like frame rate mismatches between annotation files and source videos.

## Features

- **Direct Conversion**: Converts CSV annotations (bounding boxes) into YOLO-format `.txt` files.
- **Frame Extraction**: Extracts the corresponding frames from the video and saves them as `.jpg` images.
- **Batch Processing**: Automatically finds and processes multiple video sessions from a structured directory.
- **Line-Based Sampling**: Allows you to create a smaller, evenly distributed dataset by sampling every Nth annotation line from your CSV file.
- **FPS Mismatch Handling**: Automatically detects and corrects for discrepancies in frame rates between the annotation file and the source video, ensuring your labels align with the correct frames.

## Prerequisites

You need Python 3 installed, along with the following libraries. You can install them using `pip`:

```bash
pip install pandas opencv-python
````

## Folder Structure for Batch Processing

To use the `batch_process_videos.py` script, your data must be organized in the following structure.
The main directory (`Train/` in this example) should contain a separate subfolder for each video session.

Inside each session folder, the script expects to find the video file and a subfolder (e.g., `auxiliary/`) that contains the corresponding CSV annotation file.

```
/path/to/your/project/
├── Train/
│   ├── Session_1_Video/
│   │   ├── session_1_video.mp4
│   │   └── auxiliary/
│   │       └── session_1_annotations.csv
│   │
│   ├── Session_2_Video/
│   │   ├── session_2_video.mp4
│   │   └── auxiliary/
│   │       └── session_2_annotations.csv
│   │
│   └── Session_3_Video/
│       ├── video_for_session_3.mov
│       └── some_other_folder_name/
│           └── annotations.csv
│
├── AnnotatedVideoToYOLO.py
├── batch_process_videos.py
└── README.md
```

## Usage

There are two ways to use this tool: processing a single video or batch processing multiple videos.

### 1. Processing a Single Video

Use the `AnnotatedVideoToYOLO.py` script for converting a single video and its annotation file.

#### Example Command

This command processes a single video, samples one frame for every 10 lines in the CSV, and saves the output to a specified directory.

```bash
python AnnotatedVideoToYOLO.py \
  --csv_file /path/to/your/annotations.csv \
  --video_file /path/to/your/video.mp4 \
  --output_dir /path/to/your/output_yolo_dataset \
  --line_sampling_interval 10
```

### 2. Batch Processing Multiple Videos

Use the `batch_process_videos.py` script to automatically process all video sessions located in a parent directory.

#### Example Command

This command scans the `/path/to/FishTrack23/Train` directory, processes each session folder it finds, samples every 10th annotation line, and places all the resulting YOLO datasets into the `master_yolo_output/` directory.

```bash
python batch_process_videos.py \
  --train_dir /path/to/FishTrack23/Train \
  --master_output_dir /path/to/master_yolo_output \
  --line_sampling_interval 10
```

## Script Descriptions

* **AnnotatedVideoToYOLO.py**:
  The core conversion script. It takes a single video and a CSV file as input, handles FPS correction, performs sampling, and generates the final YOLO-formatted images and label files.

* **batch\_process\_videos.py**:
  A wrapper script that automates the process for multiple videos. It iterates through a directory, finds matching video/CSV pairs, and calls `AnnotatedVideoToYOLO.py` for each one.

## Important Note: Frame Rate (FPS) Mismatch

This tool is designed to handle a common issue where annotations are created on a video with one frame rate (e.g., 10 FPS), but the source video file has a different frame rate (e.g., 30 FPS).

The script will automatically detect this mismatch by comparing the FPS value in the CSV metadata with the FPS of the video file.
If a discrepancy is found, it will adjust the frame numbers from the CSV to ensure the annotations are mapped to the correct moment in time in the source video, preventing empty or misaligned labels.

