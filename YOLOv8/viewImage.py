import cv2
import os

def view_yolo_dataset(images_dir, labels_dir):
    # 1. Get a list of all images in the folder
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return

    print(f"Found {len(image_files)} images. Controls:")
    print("  [d] - Next image")
    print("  [a] - Previous image")
    print("  [q] - Quit viewer")

    index = 0
    while True:
        # 2. Get the current image and its matching label file
        img_file = image_files[index]
        img_path = os.path.join(images_dir, img_file)
        
        # Swaps the .jpg/.png extension for .txt to find the label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        # 3. Load the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
            
        img_h, img_w, _ = img.shape

        # 4. Read labels and draw boxes (Your existing logic)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:5])

                    x1 = int((x_center - w / 2) * img_w)
                    y1 = int((y_center - h / 2) * img_h)
                    x2 = int((x_center + w / 2) * img_w)
                    y2 = int((y_center + h / 2) * img_h)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"Class: {class_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 5. Add a counter to the top corner (e.g., "Image 5/100")
        cv2.putText(img, f"{index + 1}/{len(image_files)}: {img_file}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 6. Display and wait for keyboard input
        cv2.imshow('YOLO Dataset Viewer', img)
        key = cv2.waitKey(0) & 0xFF

        # 7. Handle navigation
        if key == ord('q'):  # Quit
            break
        elif key == ord('d'):  # Next
            index = (index + 1) % len(image_files)
        elif key == ord('a'):  # Previous
            index = (index - 1) % len(image_files)

    cv2.destroyAllWindows()

# Run the function pointing to your directories
view_yolo_dataset(r"test\images", r"test\labels")