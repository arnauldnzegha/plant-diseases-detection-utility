import os
from PIL import Image, ImageDraw, ImageFont

def to_rgb_int(color_tuple):
    """Converts a color tuple (0-1 float) to an RGB integer tuple (0-255)."""
    return tuple(int(c * 255) for c in color_tuple[:3]) # Only take RGB, ignore alpha if present

def visualize_image_annotations(image_path, label_path, output_path, class_names, class_colors_dict):
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size

        # Load a default font for drawing text
        try:
            font = ImageFont.truetype("arial.ttf", 20) # Increased font size to 20
        except IOError:
            font = ImageFont.load_default() # Fallback to default pillow font

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(float(parts[0])) # Convert to float first to handle 'X.0'
                        x_center, y_center, width, height = map(float, parts[1:])

                        # Convert YOLO format to pixel coordinates (top-left, bottom-right)
                        x1 = int((x_center - width / 2) * img_width)
                        y1 = int((y_center - height / 2) * img_height)
                        x2 = int((x_center + width / 2) * img_width)
                        y2 = int((y_center + height / 2) * img_height)

                        # Get class name and color
                        if class_id < len(class_names):
                            class_name = class_names[class_id]
                            color_rgb_float = class_colors_dict.get(class_name, (0.0, 0.0, 0.0)) # Default to black if not found
                            box_color = to_rgb_int(color_rgb_float)
                        else:
                            print(f"Warning: Class ID {class_id} out of range for image {os.path.basename(image_path)}")
                            box_color = (255, 0, 0) # Default to red for unknown classes
                            class_name = f"UNKNOWN_{class_id}"

                        # Draw bounding box
                        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3) # Increased width to 3

                        # Draw class name text with background
                        text = class_name

                        bbox_text = draw.textbbox((0, 0), text, font=font)
                        text_width = bbox_text[2] - bbox_text[0]
                        text_height = bbox_text[3] - bbox_text[1]

                        text_x = x1
                        text_y = y1 - text_height - 5 # 5 pixels padding above the box

                        # Ensure text is not drawn outside image top bound
                        if text_y < 0:
                            text_y = y2 + 5 # Draw below if no space above

                        # Draw background for text
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill=box_color)

                        # Determine contrasting text color (white for dark backgrounds, black for light)
                        brightness = (box_color[0] * 299 + box_color[1] * 587 + box_color[2] * 114) / 1000
                        text_fill_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)

                        draw.text((text_x, text_y), text, fill=text_fill_color, font=font)
        else:
            print(f"No label file found for {os.path.basename(image_path)}")

        img.save(output_path)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def visualize_yolo_annotations(images_dir = '',labels_dir = '',output_dir = '', class_names=[], class_colors=[]):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Processing {len(image_files)} images...")
    for img_file in image_files:
        image_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        output_path = os.path.join(output_dir, img_file)
        # Pass the class_colors dictionary to the visualization function
        visualize_image_annotations(image_path, label_path, output_path, class_names, class_colors)

    print(f"Annotated images saved to: {output_dir}")