from models.classifier import build_classifier
from data.preprocess import load_and_preprocess_data

def evaluate_classifier():
    _, (x_test, y_test) = load_and_preprocess_data()
    classifier = build_classifier()
    classifier.load_weights('path_to_weights')
    loss, accuracy = classifier.evaluate(x_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":
    evaluate_classifier()
