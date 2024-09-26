# train_model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def train_and_save_model():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save the model using joblib
    joblib.dump(model, 'model.joblib')

    return accuracy

if __name__ == "__main__":
    train_and_save_model()