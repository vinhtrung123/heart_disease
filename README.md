# Heart Disease Prediction Application - https://predict-heart-disease2-4zyejtaesjbmaxzs7hyxtq.streamlit.app/

## Overview

This application predicts the likelihood of heart disease using a machine learning model. It is built with Python and utilizes Tkinter for the graphical user interface. The application loads a pre-trained TensorFlow model to perform predictions based on user input.

## Requirements

- **Python**: Ensure you have Python 3.x installed on your system.
- **Libraries**: The following Python libraries are required:
  - Tkinter
  - NumPy
  - Pandas
  - Scikit-learn
  - TensorFlow
  - Matplotlib

## Installation

1. **Clone the Repository**: Clone the project repository to your local machine.

   git clone <repository-url>
   cd <repository-directory>

2. **Install Required Libraries**: Install the necessary Python libraries using pip.

   pip install numpy pandas scikit-learn tensorflow matplotlib

3. **Prepare the Dataset**: Ensure the `heart.csv` dataset file is present in the same directory as the script.

4. **Load the Pre-trained Model**: Ensure the pre-trained TensorFlow model is saved and can be loaded by the script.

## Running the Application

- Execute the `dudoanbenhtim.py` script to start the application.

  python dudoanbenhtim.py

- The application will launch a GUI window where users can input relevant features to predict the risk of heart disease.

## Features

- **User-Friendly Interface**: Easy-to-use interface for inputting patient data.
- **Data Standardization**: Automatically standardizes input data using Scikit-learn's `StandardScaler`.
- **Machine Learning Model**: Uses a pre-trained TensorFlow model for predictions.
- **Visualization**: Displays prediction results and possible visualizations using Matplotlib.

## How to Use

1. Launch the application.
2. Input the required patient data into the provided fields.
3. Click the "Predict" button to see the prediction result.
4. Visualizations and detailed results will be displayed within the application.

## Customization

- Modify the `heart.csv` file to update the dataset if necessary.
- Replace the model loading code with a different TensorFlow model if needed.
