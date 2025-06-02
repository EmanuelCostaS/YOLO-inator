import os
import subprocess
import argparse
import sys

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']
CSV_EXTENSIONS = ['.csv']

def find_file_in_dir(directory, extensions):
    """Finds the first file with one of the given extensions in a directory."""
    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, item)):
            for ext in extensions:
                if item.lower().endswith(ext):
                    return os.path.join(directory, item)
    return None

def find_csv_in_subfolders(parent_directory, extensions):
    """
    Finds the first CSV file within an immediate subfolder of parent_directory.
    Searches one level deep.
    """
    for item in os.listdir(parent_directory):
        sub_path = os.path.join(parent_directory, item)
        if os.path.isdir(sub_path):
            # Found a subfolder, now look for CSV inside it
            csv_file = find_file_in_dir(sub_path, extensions)
            if csv_file:
                return csv_file
    # As a fallback, check the parent_directory itself if no CSV was found in subfolders
    # This handles cases where the CSV might not be in a dedicated subfolder
    csv_file_in_parent = find_file_in_dir(parent_directory, extensions)
    if csv_file_in_parent:
        print(f"Note: CSV found directly in {parent_directory}, not in a subfolder as primarily expected.")
        return csv_file_in_parent
    return None


def main(args):
    base_train_dir = args.train_dir
    master_output_dir = args.master_output_dir
    yolo_script_path = args.yolo_script_path

    if not os.path.isdir(base_train_dir):
        print(f"Error: Training directory '{base_train_dir}' not found.")
        return

    if not os.path.isfile(yolo_script_path):
        print(f"Error: YOLO conversion script '{yolo_script_path}' not found.")
        return

    os.makedirs(master_output_dir, exist_ok=True)

    session_folders_count = 0
    processed_count = 0
    skipped_count = 0

    for session_folder_name in os.listdir(base_train_dir):
        session_folder_path = os.path.join(base_train_dir, session_folder_name)

        if not os.path.isdir(session_folder_path):
            continue # Skip files, process only directories

        session_folders_count += 1
        print(f"\n--- Processing session folder: {session_folder_name} ---")

        video_file = find_file_in_dir(session_folder_path, VIDEO_EXTENSIONS)
        # For CSV, search within subfolders of the session_folder_path
        csv_file = find_csv_in_subfolders(session_folder_path, CSV_EXTENSIONS)


        if not video_file:
            print(f"Warning: No video file found in {session_folder_path}. Skipping.")
            skipped_count += 1
            continue

        if not csv_file:
            print(f"Warning: No CSV file found in subfolders of {session_folder_path} (or directly within it). Skipping.")
            skipped_count += 1
            continue

        print(f"  Found Video: {video_file}")
        print(f"  Found CSV:   {csv_file}")

        # Create a unique output directory for this session's YOLO data
        session_yolo_output_dir = os.path.join(master_output_dir, f"{session_folder_name}_yolo_dataset")
        os.makedirs(session_yolo_output_dir, exist_ok=True)
        print(f"  Outputting YOLO data to: {session_yolo_output_dir}")

        # Construct the command to run AnnotatedVideoToYOLO.py
        command = [
            sys.executable, # Use the current Python interpreter
            yolo_script_path,
            "--csv_file", csv_file,
            "--video_file", video_file,
            "--output_dir", session_yolo_output_dir,
            # Pass through other arguments
            "--max_gap_seconds", str(args.max_gap_seconds),
            "--padding_seconds", str(args.padding_seconds),
            "--trim_tool", args.trim_tool,
            "--image_dir_name", args.image_dir_name,
            "--label_dir_name", args.label_dir_name,
            "--train_subdir", args.train_subdir,
            "--frame_sampling_interval", str(args.frame_sampling_interval)
        ]
        if args.fps_from_video:
            command.append("--fps_from_video")
        
        # Add other boolean flags from AnnotatedVideoToYOLO.py if needed
        # For example, if AnnotatedVideoToYOLO.py had --use_feature_x:
        # if args.use_feature_x:
        # command.append("--use_feature_x")

        print(f"  Running command: {' '.join(command)}")

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                print(f"  Successfully processed {session_folder_name}.")
                print("  Output from script:")
                # Limit printing stdout/stderr to avoid overly verbose logs for successful runs
                # print(stdout) # Uncomment if you need full stdout for successful runs
                processed_count +=1
            else:
                print(f"  Error processing {session_folder_name}. Return code: {process.returncode}")
                print("  Stdout:")
                print(stdout)
                print("  Stderr:")
                print(stderr)
                skipped_count += 1
        except Exception as e:
            print(f"  An exception occurred while trying to run AnnotatedVideoToYOLO.py for {session_folder_name}: {e}")
            skipped_count += 1

    print("\n--- Batch Processing Summary ---")
    print(f"Total session folders found: {session_folders_count}")
    print(f"Successfully processed:      {processed_count}")
    print(f"Skipped due to errors/missing files: {skipped_count}")
    print(f"Master output directory: {os.path.abspath(master_output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process video and CSV annotations for YOLO training using AnnotatedVideoToYOLO.py.")
    parser.add_argument("--train_dir", required=True,
                        help="Path to the base training directory (e.g., Fishtrack23/Train) containing session folders.")
    parser.add_argument("--master_output_dir", required=True,
                        help="Path to the master directory where all YOLO datasets will be created (each in its own subfolder).")
    parser.add_argument("--yolo_script_path", default="AnnotatedVideoToYOLO.py",
                        help="Path to the AnnotatedVideoToYOLO.py script (default: assumes it's in the current directory).")

    # Arguments to pass through to AnnotatedVideoToYOLO.py
    # These should match the defaults or allow configuration as in your original script
    # Refer to AnnotatedVideoToYOLO.py for its default values
    original_script_defaults = {
        "fps_from_video": False, # This is a store_true action, handled specially
        "max_gap_seconds": 1.0,
        "padding_seconds": 0.5,
        "trim_tool": 'ffmpeg',
        "image_dir_name": 'images',
        "label_dir_name": 'labels',
        "train_subdir": 'train',
        "frame_sampling_interval": 1
    }

    parser.add_argument("--fps_from_video", action='store_true',
                        default=original_script_defaults["fps_from_video"],
                        help="If set, use FPS from video file (passed to YOLO script).")
    parser.add_argument("--max_gap_seconds", type=float,
                        default=original_script_defaults["max_gap_seconds"],
                        help="Max gap seconds (passed to YOLO script).")
    parser.add_argument("--padding_seconds", type=float,
                        default=original_script_defaults["padding_seconds"],
                        help="Padding seconds (passed to YOLO script).")
    parser.add_argument("--trim_tool",
                        default=original_script_defaults["trim_tool"], choices=['ffmpeg', 'moviepy'],
                        help="Trim tool (passed to YOLO script).")
    parser.add_argument("--image_dir_name",
                        default=original_script_defaults["image_dir_name"],
                        help="Image directory name (passed to YOLO script).")
    parser.add_argument("--label_dir_name",
                        default=original_script_defaults["label_dir_name"],
                        help="Label directory name (passed to YOLO script).")
    parser.add_argument("--train_subdir",
                        default=original_script_defaults["train_subdir"],
                        help="Training subdirectory name (passed to YOLO script).")
    parser.add_argument("--frame_sampling_interval", type=int,
                        default=original_script_defaults["frame_sampling_interval"],
                        help="Frame sampling interval (passed to YOLO script).")

    batch_args = parser.parse_args()
    main(batch_args)