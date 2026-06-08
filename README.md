# Iris Classifier (Decision Tree)

## Overview

This project is an end-to-end machine learning example that trains a Decision Tree classifier on the classic Iris flower dataset using scikit-learn.

The model predicts the species of an Iris flower based on four measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

The possible species are:

- Setosa
- Versicolor
- Virginica

## Project Structure

```text
iris-classifier/
├── src/
│   └── train.py
├── outputs/
│   └── confusion_matrix.png
├── requirements.txt
└── README.md
```

## Setup Instructions

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Longe-J/iris-classifier.git
cd iris-classifier
```


### 2. Create a Virtual Environment

On Windows:
 
```bash
python -m venv venv
```

On macOS/Linux:

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Training Script

```bash
python src/train.py --test-size 0.2 --random-state 42
```

### 6. View the Output

After the script runs, it will print the model accuracy in the terminal.

It will also create an `outputs` folder if it does not already exist and save the confusion matrix image here:

```text
outputs/confusion_matrix.png
```

## How It Works

The project follows a standard supervised machine learning workflow:

1. Load the Iris dataset
2. Split the data into training and testing sets
3. Train a Decision Tree classifier
4. Make predictions on unseen test data
5. Evaluate the model using accuracy
6. Save a confusion matrix image to the `outputs` folder

## Command Line Arguments

The training script accepts two optional command line arguments.

### `--test-size`

Controls how much of the dataset is used for testing.

Example:

```bash
python src/train.py --test-size 0.3
```

This uses 30% of the dataset for testing and 70% for training.

### `--random-state`

Controls the random seed so results can be reproduced.

Example:

```bash
python src/train.py --random-state 42
```

Using the same random state should give the same train/test split each time.

## Example Output

After running the script, the terminal should print the model accuracy, for example:

```text
Accuracy: 1.00
Saved confusion matrix to outputs/confusion_matrix.png
```

## Model Used

This project uses a `DecisionTreeClassifier` from scikit-learn.

A decision tree learns a series of rules from the training data. For example, it may learn that flowers with a small petal length are likely to be Setosa. It then uses these learned rules to predict the species of new flowers.

## Evaluation

The model is evaluated using:

- Accuracy score
- Confusion matrix

Accuracy measures the proportion of correct predictions.

The confusion matrix shows how many flowers were correctly or incorrectly classified for each species.

## Requirements

The main Python packages used are:

```text
scikit-learn
matplotlib
```

These should be listed in `requirements.txt`.

## Notes

The Iris dataset is small and clean, so the model may achieve very high accuracy, sometimes even 100%, depending on the train/test split.

This does not mean the model is perfect in every real-world situation. It means it performed well on this particular test set.