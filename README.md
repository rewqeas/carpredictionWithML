
# Car Price Prediction using TensorFlow

This project aims to predict car prices based on various features like years, mileage, rating, condition, economy, top speed, horsepower, and torque using a deep learning model built with TensorFlow.

## Project Overview

The project follows these steps:

1.  **Data Loading and Preparation:**
    *   Loads car data from a CSV file (`train.csv`).
    *   Converts the data to TensorFlow tensors.
    *   Shuffles the data to ensure unbiased training.
    *   Splits the data into training, validation, and test sets.
    *   Applies normalization to the input features.

2.  **Model Building:**
    *   Creates a sequential neural network model using TensorFlow Keras.
    *   The model includes an input layer, a normalization layer, multiple dense layers with ReLU activation, and an output layer.

3.  **Model Training:**
    *   Compiles the model with the Adam optimizer and Mean Absolute Error loss.
    *   Trains the model using the training data and validates it with the validation data for a specified number of epochs.

4.  **Model Evaluation:**
    *   Plots the training and validation loss over epochs to visualize the model's learning progress.
    *   Plots the training and validation Root Mean Squared Error (RMSE) over epochs to assess model performance.

5.  **Prediction and Visualization:**
    *   Uses the trained model to make predictions on the test data.
    *   Compares the predicted prices with the actual prices using a bar chart.

## Requirements

*   Python 3.x
*   TensorFlow
*   Pandas
*   NumPy
*   Seaborn
*   Matplotlib

## Usage

1.  Make sure you have the `train.csv` file containing your car data in the same directory as the notebook.
2.  Run the notebook cells sequentially.

## Data Description

The `train.csv` file should contain the following columns:

*   `years`: Age of the car.
*   `km`: Kilometers driven.
*   `rating`: Car's rating.
*   `condition`: Car's condition.
*   `economy`: Fuel economy.
*   `top speed`: Maximum speed.
*   `hp`: Horsepower.
*   `torque`: Torque.
*   `current price`: The price of the car (target variable).

## Code Structure

*   **Imports:** Imports necessary libraries.
*   **Data Loading and Preparation:** Loads, preprocesses, and splits the data.
*   **Model Definition:** Defines the neural network architecture.
*   **Model Training:** Compiles and trains the model.
*   **Evaluation and Visualization:** Plots training history and compares predictions with actual values.

## Author
rewqeas
