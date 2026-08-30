# NBA Star Face Classifier (End-to-End Deep Learning)

**Live Demo:** https://nba-star-face-classifier-9irsismk5lzlrqigmkcvbz.streamlit.app/

A complete End-to-End deep learning project that classifies NBA stars (LeBron James, Luka Doncic, Shai Gilgeous-Alexander) using Transfer Learning and Face Detection.

## System Architecture
To prevent the model from learning irrelevant background features (e.g., jersey colors, basketball courts), this project implements a **Two-Stage Pipeline**:
* **Face Detection (MTCNN):** Automatically detects faces in uploaded images and applies a 30% padding to preserve jawlines and hairstyles.
* **Feature Extraction & Classification (MobileNetV2):** A fine-tuned MobileNetV2 architecture with custom top layers to classify the cropped facial features.

## Tech Stack
* **Deep Learning Framework:** TensorFlow / Keras
* **Computer Vision:** MTCNN, OpenCV, PIL
* **Web Deployment:** Streamlit Community Cloud
* **Training Techniques:** Data Augmentation, Two-Phase Fine-Tuning, Early Stopping

## Error Analysis & Known Limitations
During testing, a classic **Open-Set Recognition** limitation was observed. When the model was given a photo of Michael Jordan (who is not in the training set), it predicted "LeBron James" with 90%+ confidence. 

**Root Cause:**
* **Feature Overlap:** The model learned that "dark skin + short hair/bald + facial hair" strongly correlates with LeBron James in our limited dataset.
* **Softmax Polarization:** The Softmax function forces the output probabilities to sum to 100%. Without sufficient "Hard Negatives" in the `Unknown` class, the model is forced to push all confidence toward the closest match.

## Future Work (Architecture Upgrade)
To completely resolve the Open-Set Recognition issue and the Softmax limitation, future iterations will move away from traditional classification. The next step is to implement **Metric Learning** (e.g., Siamese Networks with Triplet Loss) to calculate the embedding distance between faces, establishing an absolute rejection threshold for any unseen individuals.
