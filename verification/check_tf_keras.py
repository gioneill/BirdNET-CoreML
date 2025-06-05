try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
    print("tf.keras version:", tf.keras.__version__)
    print("tf.keras available:", hasattr(tf, "keras"))
except Exception as e:
    print("Error importing tf.keras:", e)
