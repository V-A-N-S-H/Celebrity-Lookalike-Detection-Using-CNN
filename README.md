# Celebrity Lookalike Detection Using CNN

A deep learning-based web application that detects the celebrity who most closely resembles a given face. The project uses a **Convolutional Neural Network (CNN)** for feature extraction and computes similarity between facial embeddings to recommend the closest celebrity match. The application is built using **Python** and **Streamlit**.

## Live Demo

**Deployment:**  
https://celebrity-lookalike-detection-using-cnn-qu9jzcnjgjftpqzzoszfww.streamlit.app/

---

## Features

- Upload an image and detect the closest celebrity lookalike.
- Face detection using MTCNN.
- Feature extraction using a pre-trained CNN model.
- Computes facial similarity using embeddings.
- Displays the predicted celebrity along with confidence/similarity score.
- Interactive and user-friendly Streamlit interface.

---

## Technologies Used

- Python
- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- MTCNN
- OpenCV
- NumPy
- Streamlit
- Scikit-learn
- Pillow

---

## Project Structure

```text
Celebrity-Lookalike-Detection-Using-CNN/
│
├── app.py                     # Streamlit application
├── main.ipynb                 # Model development and experimentation
├── embedding.pkl              # Celebrity feature embeddings
├── filenames.pkl              # Celebrity image paths
├── model/                     # Pre-trained CNN model
├── celebrities/               # Celebrity image dataset
├── requirements.txt           # Project dependencies
├── .gitignore
└── README.md
```

---

## Dataset

The project uses a dataset containing images of multiple celebrities. Each image is processed to extract facial features using a pre-trained CNN model.

The dataset includes:

- Celebrity Images
- Face Embeddings
- Image Paths

---

## How It Works

1. Upload an image through the Streamlit interface.
2. Detect the face using MTCNN.
3. Preprocess the detected face.
4. Extract feature embeddings using a pre-trained CNN.
5. Compare the embeddings with stored celebrity embeddings.
6. Find the closest match using similarity metrics.
7. Display the predicted celebrity.

---

## Machine Learning Workflow

- Face Detection
- Image Preprocessing
- Feature Extraction using CNN
- Embedding Generation
- Similarity Search
- Celebrity Prediction

---

## Installation

### Clone the repository

```bash
git clone https://github.com/your-username/Celebrity-Lookalike-Detection-Using-CNN.git
```

### Navigate to the project directory

```bash
cd Celebrity-Lookalike-Detection-Using-CNN
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Main libraries used:

- tensorflow
- keras
- streamlit
- numpy
- opencv-python
- mtcnn
- pillow
- scikit-learn

---

## Future Improvements

- Support multiple face detection in a single image.
- Increase the number of celebrities in the dataset.
- Improve prediction accuracy with fine-tuned CNN models.
- Add top-5 celebrity recommendations.
- Display similarity percentage and confidence score.
- Deploy using Docker and cloud services.

---

## Author

**Vansh Kashyap**

B.Tech Computer Science and Engineering

Passionate about Data Science, Machine Learning, Deep Learning, Computer Vision, and Artificial Intelligence.

---

## License

This project is intended for educational and learning purposes.
