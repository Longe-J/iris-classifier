import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    parser = argparse.ArgumentParser(description="Train a Decision Tree on the Iris dataset.")

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to use for testing. Example: 0.2 means 20%%."
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used to make the train/test split reproducible."
    )

    args = parser.parse_args()

    # Load the Iris dataset
    iris = load_iris()

    # X = input features, y = correct labels
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Create the model
    model = DecisionTreeClassifier(random_state=args.random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on unseen test data
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Create the outputs folder if it does not already exist
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Create and save the confusion matrix image
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=iris.target_names
    )

    display.plot()
    plt.title("Iris Decision Tree Confusion Matrix")
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    print("Saved confusion matrix to outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()