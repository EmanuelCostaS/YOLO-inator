import pandas as pd
import cv2
import os
import subprocess
import argparse
from datetime import datetime, timedelta

# NEW: Import MoviePy and imageio_ffmpeg
try:
    from moviepy import *
    import imageio_ffmpeg
    # Attempt to set MoviePy's FFMPEG_BINARY to the one provided by imageio_ffmpeg
    # This makes MoviePy less reliant on a system-wide FFmpeg installation.
    os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe() # For some MoviePy versions/setups
    # For newer moviepy versions, direct assignment might be needed if the above doesn't work or if moviepy.config is used.
    # import moviepy.config as  moviepy_config
    # moviepy_config.FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()
    # The most common way is to let moviepy find it, or it uses imageio_ffmpeg automatically.
    # If direct assignment is needed, it's usually:
    # from moviepy.config import change_settings
    # change_settings({"FFMPEG_BINARY": imageio_ffmpeg.get_ffmpeg_exe()})
    print(f"INFO: MoviePy will attempt to use FFmpeg from: {imageio_ffmpeg.get_ffmpeg_exe()}")
except ImportError:
    print("WARNING: MoviePy or imageio_ffmpeg not installed. MoviePy trimming will not be available or might rely on system FFmpeg.")
    VideoFileClip = None # Ensure it's defined for type hints or later checks
except Exception as e:
    print(f"WARNING: Could not configure MoviePy with imageio_ffmpeg: {e}")
    VideoFileClip = None

# --- Configuration ---
DEFAULT_FPS_FROM_VIDEO = False
DEFAULT_MAX_GAP_SECONDS = 1.0
DEFAULT_PADDING_SECONDS = 0.5
# You can change the default tool if you prefer MoviePy now
DEFAULT_TRIM_TOOL = 'ffmpeg' # or 'moviepy'
DEFAULT_OUTPUT_BASE_DIR = r'\your\path'
DEFAULT_IMAGE_DIR_NAME = 'images'
DEFAULT_LABEL_DIR_NAME = 'labels'
DEFAULT_TRAIN_SUBDIR = 'train'

# --- Helper Functions ---

def hhmmssfff_to_seconds(time_str):
    """Converts HH:MM:SS.ffffff string to total seconds."""
    if pd.isna(time_str):
        return None
    try:
        h, m, s_component = time_str.split(':')
        s_parts = s_component.split('.')
        s = float(s_parts[0])
        # ffffff represents microseconds
        microsecond = float(s_parts[1]) if len(s_parts) > 1 else 0.0
        return int(h) * 3600 + int(m) * 60 + s + (microsecond / 1e6)
    except ValueError:
        # Handle cases where milliseconds might be missing or format varies slightly
        try:
            dt_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
        except ValueError:
            try:
                dt_obj = datetime.strptime(time_str, "%H:%M:%S")
            except ValueError as e_strptime:
                print(f"Warning: Could not parse time string '{time_str}' after multiple attempts: {e_strptime}")
                return None
        return dt_obj.hour * 3600 + dt_obj.minute * 60 + dt_obj.second + dt_obj.microsecond / 1e6

def parse_csv_annotations(csv_path):
    """Parses the CSV annotation file."""
    print(f"Parsing CSV: {csv_path}")
    csv_fps = None
    try:
        # Extract FPS from metadata
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('# metadata'):
                    if 'fps:' in line:
                        try:
                            # Robustly extract FPS, handling potential extra text or delimiters
                            fps_part = line.split('fps:')[1]
                            csv_fps_str = fps_part.split(',')[0].split(';')[0].strip()
                            csv_fps = float(csv_fps_str)
                            print(f"FPS from CSV metadata: {csv_fps}")
                        except Exception as e:
                            print(f"Warning: Could not parse FPS from CSV metadata line ('{line.strip()}'): {e}")
                    # Assuming metadata related to FPS is usually near the top.
                    # If other metadata is needed, this logic might need adjustment.
                elif not line.startswith('#'): # Stop after comments if metadata is expected at the top
                    break
        
        # Columns the script expects and their corresponding 0-based indices in your CSV:
        # TimestampStr (column 1), FrameID (column 2), TL_x (column 3), TL_y (column 4), 
        # BR_x (column 5), BR_y (column 6), Species (column 9)
        cols_to_use_indices = [1, 2, 3, 4, 5, 6, 9]
        script_expected_names = ['TimestampStr', 'FrameID', 'TL_x', 'TL_y', 'BR_x', 'BR_y', 'Species']

        # Define dtypes for the columns being read.
        # Crucially, TimestampStr is read as a string.
        # Other columns are read as strings first for robustness, then converted.
        df = pd.read_csv(
            csv_path,
            comment='#',
            header=None,                         # No header row in the data part of the CSV
            usecols=cols_to_use_indices,         # Select specific columns by their index
            names=script_expected_names,         # Assign expected names to these selected columns
            dtype={                              # Define data types for these named columns
                'TimestampStr': str,
                'FrameID': str, 
                'TL_x': str,    
                'TL_y': str,
                'BR_x': str,
                'BR_y': str,
                'Species': str
            },
            skipinitialspace=True                # Handles potential spaces after delimiters
        )
        
        # print("CSV loaded into DataFrame. Initial info:")
        # df.info()
        # print("Head of DataFrame (first 5 rows):")
        # print(df.head())
            
        df['TimestampSec'] = df['TimestampStr'].apply(hhmmssfff_to_seconds)
        
        null_timestamp_count = df['TimestampSec'].isnull().sum()
        if null_timestamp_count > 0:
            print(f"\nWarning: {null_timestamp_count} rows out of {len(df)} had issues converting TimestampStr to TimestampSec.")
            # print("Rows with TimestampStr that failed conversion (showing up to 5):")
            # print(df[df['TimestampSec'].isnull()][['TimestampStr', 'TimestampSec']].head())
        
        df = df.dropna(subset=['TimestampSec']) # Drop rows where timestamp conversion failed
        df = df.sort_values(by='TimestampSec').reset_index(drop=True)

        # Convert FrameID and bounding box coordinates to numeric types
        # FrameID to nullable integer
        df['FrameID'] = pd.to_numeric(df['FrameID'], errors='coerce').astype('Int64')

        bbox_cols = ['TL_x', 'TL_y', 'BR_x', 'BR_y']
        for col in bbox_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce') 

        # Check for NaNs introduced by pd.to_numeric in bbox_cols before dropping
        nan_bbox_rows = df[bbox_cols].isnull().any(axis=1).sum()
        if nan_bbox_rows > 0:
            print(f"\nWarning: {nan_bbox_rows} rows had non-numeric bounding box coordinates that were converted to NaN and will be dropped.")
            # Consider logging these rows if inspection is needed:
            # print(df[df[bbox_cols].isnull().any(axis=1)][bbox_cols].head())
        
        # Drop rows if FrameID or any bbox coordinate is NaN after conversion
        df = df.dropna(subset=bbox_cols + ['FrameID'])
        for col in bbox_cols: # Convert bbox columns to nullable integers
            df[col] = df[col].astype('Int64')

        if df.empty:
            print("Warning: DataFrame is empty after processing and dropping NA/invalid values.")
        else:
            print(f"Successfully parsed and processed {len(df)} annotations.")
            # print("Final DataFrame info:")
            # df.info()
            # print("Final DataFrame head (first 5 rows):")
            # print(df.head())
            
        return df, csv_fps
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return pd.DataFrame(), None
    except pd.errors.EmptyDataError:
        print(f"Error: The file {csv_path} is empty or contains only comments after the header.")
        return pd.DataFrame(), None
    except Exception as e:
        print(f"Critical error occurred in parse_csv_annotations: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(), None

def get_video_properties(video_path):
    """Gets video properties using OpenCV."""
    print(f"Getting video properties for: {video_path}") # DIAGNOSTIC
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None, None, None # Added None for duration
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    print(f"Video properties for {video_path}: FPS={fps}, Width={frame_width}, Height={frame_height}, Duration={duration:.2f}s")
    return fps, frame_width, frame_height, duration

def get_video_segments(annotations_df, video_fps, max_gap_seconds, padding_seconds, video_duration):
    """Identifies video segments with annotations."""
    print("\n--- Identifying video segments ---") # DIAGNOSTIC
    if annotations_df.empty or 'TimestampSec' not in annotations_df.columns:
        print("No annotations or 'TimestampSec' column found to identify segments.")
        return [] # Return empty list

    annotated_times = sorted(annotations_df['TimestampSec'].unique()) # Access 'TimestampSec' column
    if not annotated_times:
        print("No unique annotated times found.")
        return [] # Return empty list

    segments = [] # Initialize as an empty list
    # Initialize with the first timestamp
    current_segment_start = annotated_times[0]
    current_segment_end = annotated_times[0]

    for t in annotated_times[1:]:
        if (t - current_segment_end) <= max_gap_seconds:
            current_segment_end = t
        else:
            print(f"Gap found: current_segment_end={current_segment_end:.2f}s, next_time={t:.2f}s, gap={(t - current_segment_end):.2f}s")
            start_padded = max(0, current_segment_start - padding_seconds)
            end_padded = min(video_duration, current_segment_end + padding_seconds + (1.0/video_fps if video_fps > 0 else 0))
            segments.append((start_padded, end_padded))
            current_segment_start = t
            current_segment_end = t
    
    # Add the last segment
    start_padded = max(0, current_segment_start - padding_seconds)
    end_padded = min(video_duration, current_segment_end + padding_seconds + (1.0/video_fps if video_fps > 0 else 0))
    segments.append((start_padded, end_padded))
    
    print(f"Identified {len(segments)} segments to trim.")
    return segments

def trim_video_segment_ffmpeg(input_video_path, output_video_path, start_sec, end_sec):
    """Trims a video segment using FFmpeg."""
    duration_sec = end_sec - start_sec
    if duration_sec <= 0:
        print(f"Skipping FFmpeg trim for {output_video_path}, invalid duration: {duration_sec:.2f}s")
        return False
    cmd = [
        'ffmpeg',
        '-ss', str(start_sec),
        '-i', input_video_path,
        '-to', str(end_sec),
        '-c:v', 'copy',
        '-c:a', 'copy',
        '-y',
        output_video_path
    ]
    print(f"Trimming with FFmpeg: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"Successfully trimmed to {output_video_path} using FFmpeg.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error trimming video with FFmpeg for {output_video_path}:")
        print("Command:", ' '.join(e.cmd))
        print("Return code:", e.returncode)
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        return False
    except FileNotFoundError:
        print("Error: FFmpeg executable not found. Please ensure it is installed and in your system's PATH for the 'ffmpeg' trim tool.")
        return False

def trim_video_segment_moviepy(input_video_path, output_video_path, start_sec, end_sec):
    """Trims a video segment using MoviePy."""
    print(f"Attempting MoviePy trim: In='{input_video_path}', Out='{output_video_path}', Start={start_sec:.2f}, End={end_sec:.2f}")
    
    if VideoFileClip is None:
        print("Error: MoviePy (VideoFileClip class) is not available. Cannot trim with MoviePy.")
        return False
    
    video = None
    sub_clip_obj = None
    try:
        video = VideoFileClip(input_video_path)

        if video is None:
            print(f"Error: MoviePy's VideoFileClip('{input_video_path}') returned None. Cannot process video.")
            return False
        
        subclip_method_to_use = None
        if hasattr(video, 'subclip'): 
            subclip_method_to_use = video.subclip
            print("INFO: Using standard 'video.subclip()' method.")
        elif hasattr(video, 'subclipped'): 
            subclip_method_to_use = video.subclipped
            print("INFO: Using non-standard 'video.subclipped()' method found in your environment.")
        else:
            print(f"ERROR: Neither 'subclip' nor 'subclipped' method found on the VideoFileClip object.")
            print(f"       Object type is: {type(video)}")
            print(f"       Attributes: {dir(video)}")
            print("       This indicates a significant issue with your MoviePy installation or compatibility.")
            return False

        actual_end_sec = min(end_sec, video.duration)

        if start_sec >= actual_end_sec:
            print(f"Skipping MoviePy trim for {output_video_path}, start time ({start_sec:.2f}s) is at or after effective end time ({actual_end_sec:.2f}s).")
            return False
            
        sub_clip_obj = subclip_method_to_use(start_sec, actual_end_sec)
        
        if sub_clip_obj is None:
            print(f"ERROR: The subclip operation returned None. Cannot write video file.")
            return False

        print(f"Subclip created. Attempting to write to '{output_video_path}'...")
        # MODIFICATION: Removed temp_audiofile=True
        sub_clip_obj.write_videofile(output_video_path,
                                   codec="libx264",
                                   audio_codec="aac",
                                   threads=os.cpu_count() or 2,
                                   logger='bar' 
                                  )
        print(f"Successfully trimmed to {output_video_path} using MoviePy.")
        return True

    except Exception as e:
        print(f"ERROR during MoviePy trimming for '{output_video_path}': {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False
    finally:
        if sub_clip_obj is not None and hasattr(sub_clip_obj, 'close'):
            try:
                sub_clip_obj.close()
            except Exception as e_close:
                print(f"Warning: Error closing MoviePy sub_clip_obj object: {e_close}")
        if video is not None and hasattr(video, 'close'):
            try:
                video.close()
            except Exception as e_close:
                print(f"Warning: Error closing MoviePy video object: {e_close}")
                
def extract_frames_and_convert_yolo(trimmed_video_path, segment_original_start_sec,
                                    original_annotations_df, video_fps_original,
                                    class_map, img_width, img_height,
                                    output_img_dir, output_label_dir, segment_idx):
    """Extracts frames, converts annotations to YOLO, and saves them."""
    cap = cv2.VideoCapture(trimmed_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open trimmed video {trimmed_video_path}")
        return

    frame_count_segment = 0
    actual_trimmed_fps = cap.get(cv2.CAP_PROP_FPS) # Get FPS of the trimmed segment
    if actual_trimmed_fps == 0: # Fallback if FPS is not readable
        print(f"Warning: Could not read FPS from {trimmed_video_path}. Using original video FPS {video_fps_original}.")
        actual_trimmed_fps = video_fps_original
    if actual_trimmed_fps == 0: # Further fallback if original FPS was also zero
         print(f"Error: FPS is zero for {trimmed_video_path}. Cannot process frames.")
         cap.release()
         return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_time_in_segment = frame_count_segment / actual_trimmed_fps
        original_video_time_sec = segment_original_start_sec + current_frame_time_in_segment
        
        time_tolerance = (1.0 / video_fps_original) / 2.0 if video_fps_original > 0 else 0.01 # Original FPS for tolerance matching
        
        # Corrected DataFrame filtering
        frame_annotations = original_annotations_df[
            (original_annotations_df['TimestampSec'] >= original_video_time_sec - time_tolerance) &
            (original_annotations_df['TimestampSec'] <= original_video_time_sec + time_tolerance)
        ]

        if not frame_annotations.empty:
            base_filename = f"segment_{segment_idx}_frame_{frame_count_segment:05d}"
            img_path = os.path.join(output_img_dir, f"{base_filename}.jpg")
            label_path = os.path.join(output_label_dir, f"{base_filename}.txt")

            # Ensure img_width and img_height are from the current frame if re-encoding happened,
            # but for -c copy, original video_width/height (img_width, img_height params) are fine.
            # If frame dimensions can change, use: h, w = frame.shape[:2] instead of img_width, img_height for normalization.
            # For simplicity, assuming -c copy preserves dimensions.
            current_frame_height, current_frame_width = frame.shape[:2]


            cv2.imwrite(img_path, frame)
            
            with open(label_path, 'w') as f_label:
                for _, ann_row in frame_annotations.iterrows():
                    # Correctly access Series elements by key
                    species = ann_row['Species']
                    if species not in class_map:
                        print(f"Warning: Species '{species}' not in class_map. Skipping annotation for {base_filename}.")
                        continue
                    class_id = class_map[species]
                    
                    tl_x, tl_y, br_x, br_y = ann_row['TL_x'], ann_row['TL_y'], ann_row['BR_x'], ann_row['BR_y']
                    
                    # Clamp coordinates to actual current frame dimensions
                    tl_x = max(0, tl_x)
                    tl_y = max(0, tl_y)
                    br_x = min(current_frame_width - 1, br_x) # Use actual frame width
                    br_y = min(current_frame_height - 1, br_y) # Use actual frame height

                    if br_x <= tl_x or br_y <= tl_y:
                        print(f"Warning: Invalid bbox coordinates after clamping for {base_filename}: ({tl_x},{tl_y},{br_x},{br_y}). Original: ({ann_row['TL_x']},{ann_row['TL_y']},{ann_row['BR_x']},{ann_row['BR_y']}). Skipping.")
                        continue

                    bbox_width = br_x - tl_x
                    bbox_height = br_y - tl_y
                    x_center = tl_x + bbox_width / 2.0
                    y_center = tl_y + bbox_height / 2.0

                    # Normalize using actual current frame dimensions
                    x_center_norm = x_center / current_frame_width
                    y_center_norm = y_center / current_frame_height
                    width_norm = bbox_width / current_frame_width
                    height_norm = bbox_height / current_frame_height
                    
                    # Ensure normalized values are within [0.0, 1.0]
                    x_center_norm = max(0.0, min(1.0, x_center_norm))
                    y_center_norm = max(0.0, min(1.0, y_center_norm))
                    width_norm = max(0.0, min(1.0, width_norm))
                    height_norm = max(0.0, min(1.0, height_norm))

                    f_label.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        frame_count_segment += 1
    
    cap.release()
    print(f"Processed frames for {trimmed_video_path}")

def create_class_mapping_and_files(all_species_names, output_dir):
    """Creates classes.txt and returns class_map and class_list."""
    unique_species = sorted(list(set(all_species_names)))
    class_map = {name: i for i, name in enumerate(unique_species)}
    class_list = unique_species # This is already a list of unique names

    classes_txt_path = os.path.join(output_dir, "classes.txt")
    with open(classes_txt_path, 'w') as f:
        for species_name in class_list:
            f.write(f"{species_name}\n")
    print(f"Created classes.txt at {classes_txt_path} with {len(class_list)} classes.")
    return class_map, class_list

def create_dataset_yaml(output_dir, image_dir_name, label_dir_name, train_subdir, class_list, nc):
    """Creates dataset.yaml file."""
    dataset_yaml_path = os.path.join(output_dir, "dataset.yaml")
    
    # Convert class_list to a string representation suitable for YAML
    class_list_str = str(class_list)

    content = f"""
path: {os.path.abspath(output_dir)}  # Root directory of the dataset
train: ./{os.path.join(image_dir_name, train_subdir)}  # Path to training images, relative to 'path'
val: ./{os.path.join(image_dir_name, train_subdir)}    # Path to validation images (using train for now)
# test: Optional path to test images

nc: {nc}  # Number of classes
names: {class_list_str} # List of class names
"""
    with open(dataset_yaml_path, 'w') as f:
        f.write(content)
    print(f"Created dataset.yaml at {dataset_yaml_path}")

def main(args):
    """Main processing pipeline."""
    base_output_dir = args.output_dir
    # Construct paths using os.path.join for cross-platform compatibility
    images_dir = os.path.join(base_output_dir, args.image_dir_name, args.train_subdir)
    labels_dir = os.path.join(base_output_dir, args.label_dir_name, args.train_subdir)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    annotations_df, csv_fps = parse_csv_annotations(args.csv_file)
    if annotations_df.empty:
        print("No annotations found or error in parsing. Exiting.")
        return

    video_fps, video_width, video_height, video_duration = get_video_properties(args.video_file)
    if video_fps is None or video_width is None or video_height is None or video_duration is None: # Check all properties
        print("Could not read video properties. Exiting.")
        return
    if video_duration == 0:
        print("Video duration is 0. Cannot process. Exiting.")
        return

    # Determine FPS to use
    if args.fps_from_video:
        actual_fps = video_fps
        print("Using FPS from video metadata.")
    elif csv_fps is not None:
        actual_fps = csv_fps
        print("Using FPS from CSV metadata.")
    else:
        actual_fps = video_fps # Fallback to video FPS if CSV FPS not available
        print("Warning: FPS not specified to be from video, and not found in CSV. Using video FPS as fallback.")

    if actual_fps == 0:
        print("Error: Effective FPS is 0. Cannot proceed with segment identification. Exiting.")
        return
    print(f"Using FPS: {actual_fps:.2f} for segment identification.")

    if 'Species' not in annotations_df.columns:
        print("Error: 'Species' column not found in annotations. Exiting.")
        return
    all_species = annotations_df['Species'].unique() # Access 'Species' column
    class_map, class_list = create_class_mapping_and_files(all_species, base_output_dir)
    num_classes = len(class_list)
    if num_classes == 0:
        print("No classes found from annotations. Exiting.")
        return

    segments = get_video_segments(annotations_df, actual_fps, args.max_gap_seconds, args.padding_seconds, video_duration)
    if not segments:
        print("No segments identified for trimming. Exiting.")
        return

    temp_trimmed_vids_dir = os.path.join(base_output_dir, "temp_trimmed_videos")
    os.makedirs(temp_trimmed_vids_dir, exist_ok=True)

    for i, (start_sec, end_sec) in enumerate(segments):
        trimmed_video_filename = f"segment_{i}.mp4" # MoviePy can output various formats, mp4 is good.
        trimmed_video_path = os.path.join(temp_trimmed_vids_dir, trimmed_video_filename)
        
        print(f"\nProcessing segment {i+1}/{len(segments)}: {start_sec:.2f}s to {end_sec:.2f}s")

        success = False
        if args.trim_tool == 'ffmpeg':
            success = trim_video_segment_ffmpeg(args.video_file, trimmed_video_path, start_sec, end_sec)
        elif args.trim_tool == 'moviepy': # MODIFIED: Call MoviePy function
            if VideoFileClip is None: # Check if MoviePy was imported successfully
                print("MoviePy trim tool selected, but MoviePy is not available. Skipping segment.")
                continue
            success = trim_video_segment_moviepy(args.video_file, trimmed_video_path, start_sec, end_sec)
        else:
            print(f"Unsupported trim tool: {args.trim_tool}")
            continue
        
        if success and os.path.exists(trimmed_video_path):
            extract_frames_and_convert_yolo(trimmed_video_path, start_sec,
                                            annotations_df, actual_fps, 
                                            class_map, video_width, video_height,
                                            images_dir, labels_dir, i)
        else:
            print(f"Skipping frame extraction for segment {i+1} due to trimming failure or file not found: {trimmed_video_path}")
            
    create_dataset_yaml(base_output_dir, args.image_dir_name, args.label_dir_name, args.train_subdir, class_list, num_classes)

    print("\nProcessing complete.")
    print(f"YOLO dataset generated at: {os.path.abspath(base_output_dir)}")
    print(f"  Images: {os.path.abspath(images_dir)}")
    print(f"  Labels: {os.path.abspath(labels_dir)}")
    print(f"  classes.txt and dataset.yaml are in {os.path.abspath(base_output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and CSV annotations for YOLO training.")
    parser.add_argument("--csv_file", required=True, help="Path to the input CSV annotation file.")
    parser.add_argument("--video_file", required=True, help="Path to the input video file.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_BASE_DIR, help="Base directory for the output YOLO dataset.")
    
    # Corrected boolean argument handling
    parser.add_argument("--fps_from_video", action='store_true',
                        help="If set, use FPS from video file. Otherwise, try CSV metadata, then video as fallback.")
    
    parser.add_argument("--max_gap_seconds", type=float, default=DEFAULT_MAX_GAP_SECONDS,
                        help="Maximum gap in seconds between detections to merge segments.")
    parser.add_argument("--padding_seconds", type=float, default=DEFAULT_PADDING_SECONDS,
                        help="Seconds of padding to add to the start and end of each segment.")
    parser.add_argument("--trim_tool", default=DEFAULT_TRIM_TOOL,
                        choices=['ffmpeg', 'moviepy'], # MODIFIED: Add 'moviepy'
                        help="Tool to use for trimming videos. 'ffmpeg' requires system FFmpeg. 'moviepy' uses the MoviePy library (which may use its own FFmpeg).")
    
    parser.add_argument("--image_dir_name", default=DEFAULT_IMAGE_DIR_NAME, help="Name of the image directory within the output.")
    parser.add_argument("--label_dir_name", default=DEFAULT_LABEL_DIR_NAME, help="Name of the label directory within the output.")
    parser.add_argument("--train_subdir", default=DEFAULT_TRAIN_SUBDIR, help="Name of the subdirectory for training data (e.g., 'train', 'val').")

    args = parser.parse_args()
    
    if not args.fps_from_video and DEFAULT_FPS_FROM_VIDEO: # If default is True and flag not set
         args.fps_from_video = True
     
    main(args)
