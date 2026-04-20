import os

def remap_yolo_labels(labels_dir):
    """
    Merges Spaghetti (0) and Stringing (1) into ID 0.
    Shifts Warp (2) down to ID 1.
    """
    if not os.path.exists(labels_dir):
        print(f"Directory not found: {labels_dir}")
        return

    txt_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    modified_count = 0

    print(f"Scanning {len(txt_files)} files in {labels_dir}...")

    for txt_file in txt_files:
        filepath = os.path.join(labels_dir, txt_file)
        
        # Read the current labels
        with open(filepath, 'r') as file:
            lines = file.readlines()

        new_lines = []
        file_changed = False

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            old_id = parts[0]
            new_id = old_id # Default to no change
            
            # --- THE REMAPPING LOGIC ---
            if old_id == "0":
                new_id = "0"  # Spaghetti stays 0
            elif old_id == "1":
                new_id = "0"  # Stringing becomes 0 (Merged)
                file_changed = True
            elif old_id == "2":
                new_id = "1"  # Warp shifts down to 1
                file_changed = True
                
            parts[0] = new_id
            new_lines.append(" ".join(parts) + "\n")

        # Overwrite the file only if we made changes
        if file_changed:
            with open(filepath, 'w') as file:
                file.writelines(new_lines)
            modified_count += 1

    print(f"Done! Updated {modified_count} files in {labels_dir}.")

# --- RUN THE SCRIPT ON ALL YOUR LABEL FOLDERS ---
# Change these to the actual paths of your local dataset
remap_yolo_labels(r"C:\NUS\Urex\YOLOv8\train\labels")
remap_yolo_labels(r"C:\NUS\Urex\YOLOv8\valid\labels")
remap_yolo_labels(r"C:\NUS\Urex\YOLOv8\test\labels")