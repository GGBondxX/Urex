from ultralytics import YOLO
import os

def evaluate_as_classifier(model, img_dir, lbl_dir, conf_threshold=0.50):
    """
    Forces YOLO to act as an Image-Level Binary Classifier to fairly 
    compare it against ResNet using standard F1, Precision, and Recall.
    """
    print(f"\n=== RUNNING CUSTOM IMAGE-LEVEL EVALUATION ===")
    print(f"Confidence Threshold strictly set to: {conf_threshold}")
    
    # Get the names of classes from the model
    class_names = model.names
    
    # Initialize score trackers for each class
    # TP: True Positive, FP: False Positive, FN: False Negative, TN: True Negative
    metrics = {class_id: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for class_id in class_names.keys()}
    
    # Get all test images
    test_images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_name in test_images:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + '.txt')
        
        # 1. READ GROUND TRUTH (What is actually in the photo)
        true_classes = set()
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    # YOLO label format: class_id x y w h
                    class_id = int(line.strip().split()[0])
                    true_classes.add(class_id)
        
        # 2. READ MODEL PREDICTION (What YOLO thinks is in the photo)
        # We pass the specific conf_threshold here
        results = model.predict(source=img_path, conf=conf_threshold, verbose=False)
        
        predicted_classes = set()
        # results[0].boxes.cls contains the class IDs of all bounding boxes it drew
        if len(results[0].boxes) > 0:
            for cls in results[0].boxes.cls:
                predicted_classes.add(int(cls.item()))
                
        # 3. CALCULATE METRICS PER CLASS
        for class_id in class_names.keys():
            is_truth = class_id in true_classes
            is_pred = class_id in predicted_classes
            
            if is_truth and is_pred:
                metrics[class_id]['TP'] += 1
            elif not is_truth and is_pred:
                metrics[class_id]['FP'] += 1
            elif is_truth and not is_pred:
                metrics[class_id]['FN'] += 1
            elif not is_truth and not is_pred:
                metrics[class_id]['TN'] += 1

    # 4. PRINT FINAL PAPER-READY RESULTS
    print("\n=== FINAL TEST SET RESULTS (IMAGE-LEVEL) ===")
    for class_id, m in metrics.items():
        name = class_names[class_id]
        
        # Prevent division by zero errors
        precision = m['TP'] / (m['TP'] + m['FP']) if (m['TP'] + m['FP']) > 0 else 0.0
        recall = m['TP'] / (m['TP'] + m['FN']) if (m['TP'] + m['FN']) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\nClass: {name.upper()}")
        print(f"  Confusion Matrix: TP:{m['TP']} | FP:{m['FP']} | FN:{m['FN']} | TN:{m['TN']}")
        print(f"  Precision: {precision * 100:.2f}%")
        print(f"  Recall:    {recall * 100:.2f}%")
        print(f"  F1-Score:  {f1_score * 100:.2f}%")


if __name__ == '__main__':
    # --- 1. SETUP ---
    model = YOLO('yolov8n.pt')  

    yaml_path = os.path.abspath('data.yaml') 

    # --- 2. TRAIN ---
    print("Starting training...")
    # Added batch=8 to prevent GPU's VRAM from overflowing
    results = model.train(data=yaml_path, epochs=50, imgsz=640, batch=8)

    print("Training complete!")

    # --- 3. LOAD BEST MODEL ---
    print("\nLoading the best model weights for evaluation...")
    # NOTE: If YOLO saves to train2 or train3, update this path!
    best_model = YOLO('runs/detect/train2/weights/best.pt')

    # --- 4. EVALUATE ---
    # We keep the standard validation for your online dataset to check basic box health
    print("\n--- Running Standard Validation Set Evaluation ---")
    val_metrics = best_model.val(data=yaml_path, split='val', batch=4)
    
    # Replace standard test evaluation with our Custom Script
    # Make sure these paths point to your local lab test data!
    test_images_dir = os.path.abspath('test/images')
    test_labels_dir = os.path.abspath('test/labels')
    
    evaluate_as_classifier(best_model, test_images_dir, test_labels_dir, conf_threshold=0.25)