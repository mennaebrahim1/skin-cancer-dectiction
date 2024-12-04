# Multiple Skin Disease Detection and Classification

This project uses deep learning to detect and classify multiple skin diseases based on dermatoscopic images. It implements a Convolutional Neural Network (CNN) using PyTorch Lightning to process the **Multiple Skin Disease Detection and Classification Dataset**, which includes diverse skin disease categories.

## Features
- **Dataset:** Contains images of 7 skin disease classes:
  - **Melanoma**
  - **Actinic Keratosis**
  - **Basal Cell Carcinoma**
  - **Dermatofibroma**
  - **Nevus**
  - **Pigmented Benign Keratosis**
  - **Seborrheic Keratosis**
  - **Squamous Cell Carcinoma**
  - **Vascular Lesion**
- **Objective:** Enhance healthcare by developing models that assist dermatologists in decision-making.
- **Preprocessing:** Data augmentation using rotation, flipping, resizing, and normalization.
- **Model Framework:** PyTorch Lightning for modular and scalable training.
- **Evaluation:** Comprehensive metrics, including precision, recall, and F1 score.

## Dataset
The dataset is sourced from the [ISIC Archive](https://www.kaggle.com/datasets/pritpal2873/multiple-skin-disease-detection-and-classification) and is hosted on Kaggle. It can be used to develop deep learning models to improve dermatological diagnostics.


## Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or an equivalent IDE
- Required Python packages (install via pip):
  ```bash
  pip install pytorch-lightning torchvision scikit-learn matplotlib
  ```

## Getting Started
1. **Clone the Repository:**  
   Clone or download this project to your local environment.
2. **Download Dataset:**  
   Download the dataset from [this link](https://www.kaggle.com/datasets/pritpal2873/multiple-skin-disease-detection-and-classification) and organize it into folders by disease category.
3. **Run the Notebook:**  
   Open the `multiple-skin-disease-detection-and-classification.ipynb` notebook in Jupyter and execute the cells step by step.

## Workflow
1. **Install Dependencies:** Install required libraries, including PyTorch Lightning.
2. **Data Loading and Augmentation:**  
   - Load the dataset.
   - Apply transformations like rotation, resizing, and normalization.
3. **Model Architecture:**  
   Design a CNN model using PyTorch Lightning.
4. **Training and Validation:**  
   Train the model and evaluate its performance using validation data.
5. **Performance Metrics:**  
   Generate metrics like classification reports to analyze precision, recall, and F1 scores.

## Results
At the end of training, the notebook outputs a detailed classification report and visualizations of the results.

## Applications
This project can be applied in:
- Healthcare diagnostics to assist dermatologists in identifying skin diseases.
- Developing automated systems for skin disease screening and triage.

## Acknowledgments
- **Dataset:** [Multiple Skin Disease Dataset](https://www.kaggle.com/datasets/pritpal2873/multiple-skin-disease-detection-and-classification)
- **Frameworks:** PyTorch, PyTorch Lightning
- **Source:** ISIC Archive for dataset and research inspiration

## License
This project is for educational and research purposes.
