import os
import csv
import shutil

# --- FULL 12-CLASS MAPPING ---
# Matches the exact index from your Roboflow data.yaml
CLASS_MAPPING = {
    0: 'spaghetti_and_stringing',
    1: 'warp',
}

def convert_yolo_to_multilabel(images_dir, labels_dir, output_dir):
    # Create output directories
    out_images_dir = os.path.join(output_dir, 'images')
    os.makedirs(out_images_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'labels.csv')

    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]
    
    print(f"Processing {len(image_files)} images for 12-Class ResNet...")

    # Prepare the CSV structure automatically
    csv_data = []
    # Header: image_name, normal, then all 12 defect classes
    defect_classes = list(CLASS_MAPPING.values())
    header = ['image_name', 'normal'] + defect_classes
    csv_data.append(header)

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        # Initialize all flags to 0
        flags = {name: 0 for name in defect_classes}
        flags['normal'] = 0

        # Check if the label file exists and has content
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_name = CLASS_MAPPING.get(class_id)
                    
                    # Flip the flag to 1 if the class exists in the file
                    if class_name in flags:
                        flags[class_name] = 1

            # If somehow a label file existed but had no recognized classes, mark as normal
            if sum([flags[name] for name in defect_classes]) == 0:
                flags['normal'] = 1
        else:
            # No label file = No defects = Normal Print
            flags['normal'] = 1

       # 1. NEW: Clean the Roboflow hash out of the filename
        if "_jpg.rf." in img_file:
            clean_img_name = img_file.split("_jpg.rf.")[0] + ".jpg"
        elif ".rf." in img_file: # Catches other extensions if they occur
            clean_img_name = img_file.split(".rf.")[0] + ".jpg"
        else:
            clean_img_name = img_file

        # 2. UPDATED: Build the row using the clean name
        row = [clean_img_name, flags['normal']] + [flags[name] for name in defect_classes]
        csv_data.append(row)

        # 3. UPDATED: Copy AND RENAME the image to the new folder
        shutil.copy2(img_path, os.path.join(out_images_dir, clean_img_name))

    # Save the CSV file
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"Done! Dataset assembled at: {output_dir}")
    print(f"CSV mapping saved to: {csv_path}")

# Run the function on your directories
convert_yolo_to_multilabel(r"train\images", r"train\labels", r"multilabel_dataset\train")
convert_yolo_to_multilabel(r"valid\images", r"valid\labels", r"multilabel_dataset\valid")