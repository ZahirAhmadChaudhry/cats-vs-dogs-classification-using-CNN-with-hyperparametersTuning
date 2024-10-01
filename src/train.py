import os
from data_loader import load_data
from model import build_model

def train_model(data_directory, img_size=(150, 150), batch_size=32, epochs=10):
    """
    Trains a CNN model on the Cats vs. Dogs dataset.

    Args:
        data_directory (str): Path to the data directory containing 'train' and 'test' folders.
        img_size (tuple): Target size of images (default is 150x150).
        batch_size (int): Number of images per batch (default is 32).
        epochs (int): Number of epochs for training (default is 10).
    """
    # Load data
    train_generator, val_generator, _ = load_data(data_directory, img_size, batch_size)

    # Build model
    model = build_model(input_shape=(img_size[0], img_size[1], 3))

    # Train the model
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    # Save the model
    model.save('cats_vs_dogs_model.keras')

if __name__ == "__main__":
    # Define the base data directory
    data_directory = 'data'

    # Train the model
    train_model(data_directory=data_directory)
