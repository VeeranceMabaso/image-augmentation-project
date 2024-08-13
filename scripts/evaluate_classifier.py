import tensorflow as tf
from data.preprocess import load_and_preprocess_data

def evaluate_classifier():
    (_, _), (x_test, y_test) = load_and_preprocess_data()
    classifier = tf.keras.models.load_model('classifier.h5')
    
    loss, accuracy = classifier.evaluate(x_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":
    evaluate_classifier()
