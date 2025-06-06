import pandas as pd
import cv2
import os
import argparse

# --- Configuration ---
DEFAULT_OUTPUT_BASE_DIR = r'your\path'  # CHANGE THIS to your default output path
DEFAULT_IMAGE_DIR_NAME = 'images'
DEFAULT_LABEL_DIR_NAME = 'labels'
DEFAULT_TRAIN_SUBDIR = 'train'
DEFAULT_LINE_SAMPLING_INTERVAL = 1 # Process every annotated line by default

# --- Helper Functions ---

def parse_csv_annotations(csv_path):
    """
    Parses the CSV annotation file.
    The main logic now sorts by and relies on FrameID.
    """
    print(f"Parsing CSV: {csv_path}")
    csv_fps = None
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('# metadata'):
                    if 'fps:' in line:
                        try:
                            fps_part = line.split('fps:')[1]
                            csv_fps_str = fps_part.split(',')[0].split(';')[0].strip()
                            csv_fps = float(csv_fps_str)
                            print(f"Found FPS in CSV metadata: {csv_fps}")
                        except Exception as e:
                            print(f"Warning: Could not parse FPS from CSV metadata line ('{line.strip()}'): {e}")
                elif not line.startswith('#'):
                    break

        cols_to_use_indices = [2, 3, 4, 5, 6, 9]
        col_names = ['FrameID', 'TL_x', 'TL_y', 'BR_x', 'BR_y', 'Species']

        df = pd.read_csv(
            csv_path,
            comment='#',
            header=None,
            usecols=cols_to_use_indices,
            names=col_names,
            dtype={name: str for name in col_names},
            skipinitialspace=True
        )

        df['FrameID'] = pd.to_numeric(df['FrameID'], errors='coerce')
        df.dropna(subset=['FrameID'], inplace=True)
        df['FrameID'] = df['FrameID'].astype('int64')

        bbox_cols = ['TL_x', 'TL_y', 'BR_x', 'BR_y']
        for col in bbox_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        nan_rows = df[bbox_cols].isnull().any(axis=1).sum()
        if nan_rows > 0:
            print(f"Warning: {nan_rows} rows had non-numeric bounding box coordinates and will be dropped.")
            df.dropna(subset=bbox_cols, inplace=True)

        for col in bbox_cols:
            df[col] = df[col].round().astype('Int64')

        # Sort by FrameID to ensure chronological processing if needed later
        df.sort_values(by='FrameID', inplace=True)
        df.reset_index(drop=True, inplace=True)

        if df.empty:
            print("Warning: DataFrame is empty after processing.")
        else:
            print(f"Successfully parsed {len(df)} total annotations.")

        return df, csv_fps
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
    except Exception as e:
        print(f"Critical error in parse_csv_annotations: {e}")
    return pd.DataFrame(), None

def get_video_properties(video_path):
    """Gets video properties using OpenCV."""
    print(f"Getting video properties for: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video properties: FPS={fps:.2f}, Resolution={width}x{height}, Total Frames={total_frames}")
    return fps, width, height, total_frames

def extract_annotated_frames(
    video_path, annotations_df, video_width, video_height,
    class_map, output_img_dir, output_label_dir):
    """
    Reads the video and saves frames specified in the (pre-sampled) annotations_df.
    """
    annotated_frame_ids = set(annotations_df['FrameID'].unique())
    print(f"\nExtracting {len(annotated_frame_ids)} unique frames from the video...")

    if not annotated_frame_ids:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_num = 0
    saved_frames_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num in annotated_frame_ids:
            frame_annotations = annotations_df[annotations_df['FrameID'] == frame_num]

            if not frame_annotations.empty:
                base_filename = f"frame_{frame_num:06d}"
                img_path = os.path.join(output_img_dir, f"{base_filename}.jpg")
                label_path = os.path.join(output_label_dir, f"{base_filename}.txt")

                # Only write the image if it doesn't exist, to prevent redundant work
                # in cases where multiple annotations are on the same sampled frame.
                if not os.path.exists(img_path):
                    cv2.imwrite(img_path, frame)
                    saved_frames_count += 1
                    if saved_frames_count % 50 == 0:
                        print(f"Saved {saved_frames_count} images...")

                with open(label_path, 'w') as f_label:
                    for _, ann_row in frame_annotations.iterrows():
                        class_id = class_map.get("Fish", 0)

                        tl_x, tl_y, br_x, br_y = ann_row['TL_x'], ann_row['TL_y'], ann_row['BR_x'], ann_row['BR_y']
                        bbox_width = br_x - tl_x
                        bbox_height = br_y - tl_y

                        if bbox_width <= 0 or bbox_height <= 0: continue

                        x_center = tl_x + bbox_width / 2.0
                        y_center = tl_y + bbox_height / 2.0

                        x_center_norm = min(1.0, max(0.0, x_center / video_width))
                        y_center_norm = min(1.0, max(0.0, y_center / video_height))
                        width_norm = min(1.0, max(0.0, bbox_width / video_width))
                        height_norm = min(1.0, max(0.0, bbox_height / video_height))

                        f_label.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

        # Stop early if all required frames have been saved
        if saved_frames_count == len(annotated_frame_ids):
            break
            
        frame_num += 1

    cap.release()
    print(f"\nFinished. Total unique images saved: {saved_frames_count}")

def create_class_mapping_and_files(output_dir):
    """Creates a fixed classes.txt for a single 'Fish' class."""
    class_list = ["Fish"]
    class_map = {"Fish": 0}
    os.makedirs(output_dir, exist_ok=True)
    classes_txt_path = os.path.join(output_dir, "classes.txt")
    with open(classes_txt_path, 'w') as f:
        for class_name in class_list:
            f.write(f"{class_name}\n")
    print(f"Created classes.txt at {classes_txt_path}")
    return class_map, class_list

def create_dataset_yaml(output_dir, image_dir_name, label_dir_name, train_subdir, class_list):
    """Creates the dataset.yaml file for YOLO."""
    dataset_yaml_path = os.path.join(output_dir, "dataset.yaml")
    content = (
        f"path: {os.path.abspath(output_dir)}\n"
        f"train: ./{os.path.join(image_dir_name, train_subdir)}\n"
        f"val: ./{os.path.join(image_dir_name, train_subdir)}\n"
        f"\nnc: {len(class_list)}\n"
        f"names: {str(class_list)}\n"
    )
    with open(dataset_yaml_path, 'w') as f:
        f.write(content)
    print(f"Created dataset.yaml at {dataset_yaml_path}")

def main(args):
    """Main processing pipeline."""
    base_output_dir = args.output_dir
    images_dir = os.path.join(base_output_dir, args.image_dir_name, args.train_subdir)
    labels_dir = os.path.join(base_output_dir, args.label_dir_name, args.train_subdir)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    annotations_df, csv_fps = parse_csv_annotations(args.csv_file)
    if annotations_df.empty:
        print("No valid annotations found. Exiting.")
        return
        
    # --- NEW LOGIC: Sample the DataFrame based on its rows (lines) ---
    if args.line_sampling_interval > 1:
        print(f"\nSampling 1 out of every {args.line_sampling_interval} annotation lines from the CSV...")
        # This selects the 0th row, Nth row, 2Nth row, etc. from the dataframe
        annotations_df = annotations_df.iloc[::args.line_sampling_interval].copy()
        annotations_df.reset_index(drop=True, inplace=True)
        print(f"Number of annotations after sampling: {len(annotations_df)}")
    # --- END OF NEW LOGIC ---

    video_fps, video_width, video_height, _ = get_video_properties(args.video_file)
    if video_fps is None:
        print("Could not read video properties. Exiting.")
        return

    if csv_fps and abs(csv_fps - video_fps) > 1:
        print(f"\nWARNING: CSV FPS ({csv_fps}) and video FPS ({video_fps}) differ.")
        print("Adjusting annotation FrameIDs to match the video's time base...")
        annotations_df['FrameID'] = (annotations_df['FrameID'] / csv_fps * video_fps).round().astype(int)

    class_map, class_list = create_class_mapping_and_files(base_output_dir)

    extract_annotated_frames(
        args.video_file, annotations_df, video_width, video_height,
        class_map, images_dir, labels_dir
    )
    
    create_dataset_yaml(base_output_dir, args.image_dir_name, args.label_dir_name, args.train_subdir, class_list)

    print("\nProcessing complete.")
    print(f"YOLO dataset generated at: {os.path.abspath(base_output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video and CSV annotations for YOLO training.")
    parser.add_argument("--csv_file", required=True, help="Path to the input CSV annotation file.")
    parser.add_argument("--video_file", required=True, help="Path to the input video file.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_BASE_DIR, help="Base directory for the output YOLO dataset.")
    parser.add_argument("--image_dir_name", default=DEFAULT_IMAGE_DIR_NAME, help="Name of the image directory.")
    parser.add_argument("--label_dir_name", default=DEFAULT_LABEL_DIR_NAME, help="Name of the label directory.")
    parser.add_argument("--train_subdir", default=DEFAULT_TRAIN_SUBDIR, help="Name of the training subdirectory.")
    parser.add_argument("--line_sampling_interval", type=int, default=DEFAULT_LINE_SAMPLING_INTERVAL, help="Sample one frame for every X lines in the CSV. Default is 1 (process every annotated line).")

    args = parser.parse_args()
    if args.line_sampling_interval < 1:
        parser.error("--line_sampling_interval must be 1 or greater.")

    main(args)
