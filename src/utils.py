import tensorflow as tf

# Check if TensorFlow can detect the GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
