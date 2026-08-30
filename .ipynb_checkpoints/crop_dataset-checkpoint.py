import os
import cv2
from mtcnn import MTCNN
import shutil

# 設定來源與目標資料夾
INPUT_DIR = 'Dataset'
OUTPUT_DIR = 'Dataset_Faces'
PADDING_RATIO = 0.3  # 往外擴張 30%，保留頭髮和下巴

# 啟動找臉專家
detector = MTCNN()

# 建立新的 Dataset_Faces 資料夾 (如果已存在就先刪除重建)
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

print("🚀 開始自動裁切人臉資料集...")

# 遍歷 Dataset 底下的 4 個分類資料夾
for class_name in os.listdir(INPUT_DIR):
    class_path = os.path.join(INPUT_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
        
    # 在新的資料夾裡也建立對應的分類
    new_class_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(new_class_path)
    
    print(f"\n處理分類: {class_name}...")
    
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        new_img_path = os.path.join(new_class_path, img_name)
        
        # 讀取圖片並轉為 RGB
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 偵測人臉
        faces = detector.detect_faces(img_rgb)
        
        if faces:
            # 抓取第一張臉的座標
            x, y, width, height = faces[0]['box']
            
            # 計算 Padding，讓臉部不會裁得太死
            pad_w = int(width * PADDING_RATIO)
            pad_h = int(height * PADDING_RATIO)
            
            y1 = max(0, y - pad_h)
            y2 = min(img.shape[0], y + height + pad_h)
            x1 = max(0, x - pad_w)
            x2 = min(img.shape[1], x + width + pad_w)
            
            # 裁切圖片 (用原本的 BGR 圖片裁切，方便直接存檔)
            cropped_face = img[y1:y2, x1:x2]
            cv2.imwrite(new_img_path, cropped_face)
        else:
            # 如果這張照片找不到人臉 (例如 Unknown 裡的動漫人物或風景)
            # 就直接把原圖複製過去，不丟失負樣本
            shutil.copy(img_path, new_img_path)
            print(f"  - 找不到人臉，保留原圖: {img_name}")

print("\n✅ 所有圖片處理完畢！請去檢查 Dataset_Faces 資料夾！")