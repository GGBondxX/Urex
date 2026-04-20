from ultralytics import YOLO
import os

def evaluate_as_classifier(model, img_dir, lbl_dir, conf_threshold=0.25):
    """
    Forces YOLO to act as an Image-Level Binary Classifier to output 
    Precision, Recall, and F1-Score.
    """
    print(f"\n=== RUNNING CUSTOM IMAGE-LEVEL EVALUATION ===")
    print(f"Confidence Threshold strictly set to: {conf_threshold}")
    
    class_names = model.names
    
    # Initialize trackers
    metrics = {class_id: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for class_id in class_names.keys()}
    metrics['NORMAL'] = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    
    test_images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(test_images)} images in {img_dir}")
    
    for img_name in test_images:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + '.txt')
        
        # 1. READ GROUND TRUTH
        true_classes = set()
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    class_id = int(line.strip().split()[0])
                    true_classes.add(class_id)
        
        # 2. READ MODEL PREDICTION
        results = model.predict(source=img_path, conf=conf_threshold, verbose=False)
        
        predicted_classes = set()
        if len(results[0].boxes) > 0:
            for cls in results[0].boxes.cls:
                predicted_classes.add(int(cls.item()))
                
        
        is_actually_normal = (len(true_classes) == 0)
        is_predicted_normal = (len(predicted_classes) == 0)

        if is_actually_normal and is_predicted_normal:
            metrics['NORMAL']['TP'] += 1
        elif not is_actually_normal and is_predicted_normal:
            metrics['NORMAL']['FP'] += 1
        elif is_actually_normal and not is_predicted_normal:
            metrics['NORMAL']['FN'] += 1
        elif not is_actually_normal and not is_predicted_normal:
            metrics['NORMAL']['TN'] += 1

        # 3. CALCULATE METRICS
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

    # 4. PRINT FINAL RESULTS
    print("\n=== FINAL TEST SET RESULTS (IMAGE-LEVEL) ===")
    for key, m in metrics.items():
        # Handle the custom Normal name, otherwise use the yaml name
        name = "NORMAL (NO DEFECTS)" if key == 'NORMAL' else class_names[key]
        
        precision = m['TP'] / (m['TP'] + m['FP']) if (m['TP'] + m['FP']) > 0 else 0.0
        recall = m['TP'] / (m['TP'] + m['FN']) if (m['TP'] + m['FN']) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\nClass: {name.upper()}")
        print(f"  Confusion Matrix: TP:{m['TP']} | FP:{m['FP']} | FN:{m['FN']} | TN:{m['TN']}")
        print(f"  Precision: {precision * 100:.2f}%")
        print(f"  Recall:    {recall * 100:.2f}%")
        print(f"  F1-Score:  {f1_score * 100:.2f}%")


if __name__ == '__main__':
    # --- 1. LOAD SPECIFIC MODEL ---
    print("Loading model from train2...")
    best_model = YOLO(os.path.abspath('runs/detect/train2/weights/best.pt'))

    # --- 2. SET YOUR NEW DATASET PATHS ---
    # Pointing directly to your new test2 folder
    test_images_dir = os.path.abspath('test2/images')
    test_labels_dir = os.path.abspath('test2/labels')
    
    # --- 3. RUN EVALUATION ---
    evaluate_as_classifier(best_model, test_images_dir, test_labels_dir, conf_threshold=0.25)