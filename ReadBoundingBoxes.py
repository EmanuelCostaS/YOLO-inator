import cv2

def draw_bounding_box(image_path, yolo_path, output_path="output.jpg"):
    """
    Draws a bounding box on an image based on YOLO format coordinates.

    Args:
        image_path (str): The path to the input image.
        yolo_path (str): The path to the YOLO format txt file.
        output_path (str, optional): The path to save the output image. 
                                     Defaults to "output.jpg".
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image at {image_path}")
        return

    # Get image dimensions
    height, width, _ = image.shape

    # Read the YOLO data from the txt file
    try:
        with open(yolo_path, 'r') as f:
            yolo_data = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find the YOLO file at {yolo_path}")
        return

    for line in yolo_data:
        # Split the line into its components
        class_id, x_center_norm, y_center_norm, width_norm, height_norm = map(float, line.split())

        # Denormalize the coordinates
        x_center = int(x_center_norm * width)
        y_center = int(y_center_norm * height)
        box_width = int(width_norm * width)
        box_height = int(height_norm * height)

        # Calculate the top-left corner of the bounding box
        x_min = int(x_center - (box_width / 2))
        y_min = int(y_center - (box_height / 2))

        # Draw the bounding box on the image
        color = (0, 255, 0)  # Green color for the box
        thickness = 2
        cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min + box_height), color, thickness)

        # Put the class label on the image
        label = f"Class: {int(class_id)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = x_min
        text_y = y_min - 10 if y_min - 10 > 10 else y_min + text_size[1] + 10
        cv2.putText(image, label, (text_x, text_y), font, font_scale, color, font_thickness)


    # Save the output image
    cv2.imwrite(output_path, image)
    print(f"Output image with bounding box saved to {output_path}")

    # Display the image (optional)
    cv2.imshow('Image with Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
        
    image_file_path = r"C:\Users\edusa\Documents\Emanuel\Artigo\ReadBoundingBox\YOLO_Dataset\images\image_20250605_144850201.png"
    yolo_file_path = r"C:\Users\edusa\Documents\Emanuel\Artigo\ReadBoundingBox\YOLO_Dataset\labels\image_20250605_144850201.txt"      
    draw_bounding_box(image_file_path, yolo_file_path)
