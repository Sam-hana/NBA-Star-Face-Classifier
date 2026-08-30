# NBA Star Face Classifier

**Live Demo:** https://nba-star-face-classifier-9irsismk5lzlrqigmkcvbz.streamlit.app/

This is an end-to-end deep learning project that classifies NBA stars (LeBron James, Luka Doncic, and Shai Gilgeous-Alexander). It uses Transfer Learning and Face Detection to achieve better accuracy.

## System Architecture
To stop the model from learning background details (like jersey colors or the basketball court), this project uses a **Two-Stage Pipeline**:
* **Face Detection (MTCNN):** Automatically finds faces in uploaded images and adds a 30% padding to keep the jawline and hairstyle.
* **Classification (MobileNetV2):** Uses a fine-tuned MobileNetV2 model to classify only the cropped face.

## Tech Stack
* **Deep Learning Framework:** TensorFlow / Keras
* **Computer Vision:** MTCNN, OpenCV, PIL
* **Web Deployment:** Streamlit Community Cloud
* **Training Techniques:** Data Augmentation, Two-Phase Fine-Tuning, Early Stopping

## Project Structure
* `crop_dataset.py`: Uses MTCNN to detect and crop faces from the raw images.
* `train.py`: The script used to train and fine-tune the MobileNetV2 model.
* `app.py`: The main script for the Streamlit web application.
* `nba_star_classifier.keras`: The trained model file.
* `requirements.txt`: A list of Python packages needed to run this project.


## Error Analysis & Limitations
During testing, I found a common machine learning problem called Open-Set Recognition. When I tested the model with a photo of Michael Jordan (who is not in the dataset), it predicted "LeBron James" with over 90% confidence.

**Why did this happen?**
* **Feature Overlap:** The model learned that "dark skin + short hair + facial hair" means LeBron James in my small dataset.
* **Softmax Limitation:** The Softmax function forces all probabilities to add up to 100%. Because I didn't have enough Hard Negatives(like other players with similar features) in the `Unknown` category, the model was forced to choose the closest match.

## Future Work
To fix this Open-Set Recognition issue, the next step is to use **Metric Learning** (like Siamese Networks with Triplet Loss) instead of standard Softmax classification. This will calculate the distance between face features, allowing the system to reject people it has never seen before easily.

## 💻 How to Run Locally
If you want to run this web app on your own computer, please follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Sam-hana/NBA-Star-Face-Classifier.git
   ```
   ```bash
   cd NBA-Star-Face-Classifier
   ```

2. Install the required packages
  ```bash
  pip install -r requirements.txt
  ```
3. Run the Streamlit app:
  ```bash
  streamlit run app.py
  ```

   
