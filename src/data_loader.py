from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_data(data_directory, img_size=(150, 150), batch_size=32):
    """
    Loads training, validation, and test data from the specified directory.

    Args:
        data_directory (str): The base directory containing 'train' and 'test' folders.
        img_size (tuple): The target size to resize images (default is 150x150).
        batch_size (int): Number of images per batch (default is 32).

    Returns:
        train_generator: Keras ImageDataGenerator for training data.
        val_generator: Keras ImageDataGenerator for validation data.
        test_generator: Keras ImageDataGenerator for test data.
    """
    train_dir = os.path.join(data_directory, 'train')
    test_dir = os.path.join(data_directory, 'test')

    # Only apply rescaling, no data augmentation
    train_val_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Create train and validation generators
    train_generator = train_val_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_val_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    # Create test generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, val_generator, test_generator
