import os
import subprocess
import argparse
import sys

VIDEO_EXTENSIONS = ['.mp4', '.avi', 'mov', '.mkv']
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
            csv_file = find_file_in_dir(sub_path, extensions)
            if csv_file:
                return csv_file
    
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
            continue

        session_folders_count += 1
        print(f"\n--- Processing session folder: {session_folder_name} ---")

        video_file = find_file_in_dir(session_folder_path, VIDEO_EXTENSIONS)
        csv_file = find_csv_in_subfolders(session_folder_path, CSV_EXTENSIONS)

        if not video_file:
            print(f"Warning: No video file found in {session_folder_path}. Skipping.")
            skipped_count += 1
            continue

        if not csv_file:
            print(f"Warning: No CSV file found in or under {session_folder_path}. Skipping.")
            skipped_count += 1
            continue

        print(f"  Found Video: {video_file}")
        print(f"  Found CSV:   {csv_file}")

        session_yolo_output_dir = os.path.join(master_output_dir, f"{session_folder_name}_yolo_dataset")
        os.makedirs(session_yolo_output_dir, exist_ok=True)
        print(f"  Outputting YOLO data to: {session_yolo_output_dir}")

        # --- COMMAND CONSTRUCTION (SIMPLIFIED) ---
        # Construct the command to run the updated AnnotatedVideoToYOLO.py
        command = [
            sys.executable,
            yolo_script_path,
            "--csv_file", csv_file,
            "--video_file", video_file,
            "--output_dir", session_yolo_output_dir,
            # Pass through the relevant arguments
            "--image_dir_name", args.image_dir_name,
            "--label_dir_name", args.label_dir_name,
            "--train_subdir", args.train_subdir,
            "--line_sampling_interval", str(args.line_sampling_interval) # Use the new argument
        ]
        # --- END OF COMMAND CONSTRUCTION ---

        print(f"  Running command: {' '.join(command)}")

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                print(f"  Successfully processed {session_folder_name}.")
                processed_count +=1
            else:
                print(f"  Error processing {session_folder_name}. Return code: {process.returncode}")
                print("  Stdout:")
                print(stdout)
                print("  Stderr:")
                print(stderr)
                skipped_count += 1
        except Exception as e:
            print(f"  An exception occurred while trying to run the script for {session_folder_name}: {e}")
            skipped_count += 1

    print("\n--- Batch Processing Summary ---")
    print(f"Total session folders found: {session_folders_count}")
    print(f"Successfully processed:      {processed_count}")
    print(f"Skipped due to errors/missing files: {skipped_count}")
    print(f"Master output directory: {os.path.abspath(master_output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process video sessions using AnnotatedVideoToYOLO.py.")
    parser.add_argument("--train_dir", required=True,
                        help="Path to the base training directory (e.g., Fishtrack23/Train).")
    parser.add_argument("--master_output_dir", required=True,
                        help="Path to the master directory where all YOLO datasets will be created.")
    parser.add_argument("--yolo_script_path", default="AnnotatedVideoToYOLO.py",
                        help="Path to the AnnotatedVideoToYOLO.py script.")

    # --- ARGUMENT PARSING (SIMPLIFIED) ---
    # Arguments to pass through to the YOLO script
    parser.add_argument("--image_dir_name", default='images', help="Image directory name (passed to YOLO script).")
    parser.add_argument("--label_dir_name", default='labels', help="Label directory name (passed to YOLO script).")
    parser.add_argument("--train_subdir", default='train', help="Training subdirectory name (passed to YOLO script).")
    
    # Updated sampling argument
    parser.add_argument("--line_sampling_interval", type=int, default=1,
                        help="Sample one frame every X lines in the CSV (passed to YOLO script).")
    # --- END OF ARGUMENT PARSING ---

    batch_args = parser.parse_args()
    main(batch_args)
