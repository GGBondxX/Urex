import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_confusion_matrix(tp, fp, fn, tn, class_name="Defect"):
    """
    Takes raw confusion matrix values and generates a publication-ready heatmap diagram.
    """
    # Standard ML orientation for binary classification:
    # Top Row: Actual Negatives (TN, FP)
    # Bottom Row: Actual Positives (FN, TP)
    matrix = np.array([[tn, fp],
                       [fn, tp]])

    # Create descriptive labels for each cell
    group_names = ['True Negative', 'False Positive', 
                   'False Negative', 'True Positive']
    
    group_counts = [f"{value}" for value in matrix.flatten()]
    
    # Combine the names and the counts into one string per cell
    labels = [f"{v1}\n\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)

    # Set up the plot size and font scale
    plt.figure(figsize=(7, 5))
    sns.set_theme(font_scale=1.1)

    # Generate the heatmap (Blues color map looks highly professional)
    ax = sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues', 
                     cbar=False, linewidths=2, linecolor='black',
                     xticklabels=['Negative', f'Positive'],
                     yticklabels=['Negative', f'Positive'])

    # Format the titles and axes
    plt.title(f'Confusion Matrix: {class_name.replace("_", " ")}', fontsize=16, pad=20, fontweight='bold')
    plt.ylabel('Ground Truth ', fontsize=12, fontweight='bold')
    plt.xlabel('Yolo Prediction', fontsize=12, fontweight='bold')

    # Ensure everything fits perfectly
    plt.tight_layout()

    # Save the diagram to your folder
    filename = f'confusion_matrix_{class_name}.png'
    plt.savefig(filename, dpi=300) # dpi=300 ensures it is high-resolution for your paper
    print(f"Diagram successfully saved as: {os.path.abspath(filename)}")
    
    # Show the plot on your screen
    plt.show()

if __name__ == '__main__':
    # --- ENTER YOUR NUMBERS HERE ---
    
    # Example 1: Your exact Spaghetti and Stringing data from earlier
    generate_confusion_matrix(tp=72, fp=28, fn=10, tn=88, class_name="Normal (No defect)")
    
    # Example 2: Your Warp data
    # generate_confusion_matrix(tp=0, fp=0, fn=85, tn=113, class_name="Warp")