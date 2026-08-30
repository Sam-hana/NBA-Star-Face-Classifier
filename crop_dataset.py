import os
import cv2
from mtcnn import MTCNN
import shutil


INPUT_DIR = 'Dataset'
OUTPUT_DIR = 'Dataset_Faces'
PADDING_RATIO = 0.3  

# Run the expert of cutting face
detector = MTCNN()

# Create a new directory, Dataset_Faces
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

print("Start cropping faces from the origin dataset...")

# Iterate all the four directory in dataset
for class_name in os.listdir(INPUT_DIR):
    class_path = os.path.join(INPUT_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    new_class_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(new_class_path)
    
    print(f"\nWorking on: {class_name}...")
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        new_img_path = os.path.join(new_class_path, img_name)
        
        # Read the image and set to RGB
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        faces = detector.detect_faces(img_rgb)
        
        if faces:
            # catch the coordinate and add the padding
            x, y, width, height = faces[0]['box']
            
            pad_w = int(width * PADDING_RATIO)
            pad_h = int(height * PADDING_RATIO)
            
            y1 = max(0, y - pad_h)
            y2 = min(img.shape[0], y + height + pad_h)
            x1 = max(0, x - pad_w)
            x2 = min(img.shape[1], x + width + pad_w)
            
            cropped_face = img[y1:y2, x1:x2]
            cv2.imwrite(new_img_path, cropped_face)
        else:
            shutil.copy(img_path, new_img_path)
            print(f"  - Cannot find human face, keep the origin image: {img_name}")

print("\nAll of the image have cropped, check the Dataset_Faces directory")
