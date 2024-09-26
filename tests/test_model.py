# test_model.py
from model.train import train_and_save_model

def test_model_accuracy():
    accuracy = train_and_save_model()
    assert accuracy > 0.9, f"Model accuracy is too low: {accuracy}"

if __name__ == "__main__":
    test_model_accuracy()