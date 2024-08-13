import tensorflow as tf
from models.classifier import build_classifier
from data.preprocess import load_and_preprocess_data

def train_classifier(epochs=10):
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    classifier = build_classifier()
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    classifier.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    classifier.save('classifier.h5')

if __name__ == "__main__":
    train_classifier()
