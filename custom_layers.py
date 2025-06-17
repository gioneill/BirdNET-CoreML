import tensorflow as tf
k = tf.keras
l = tf.keras.layers

class SimpleSpecLayer(l.Layer):

    def __init__(self, sample_rate=48000, spec_shape=(257, 384), frame_step=374, frame_length=512, data_format='channels_last', **kwargs):
        super(SimpleSpecLayer, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.spec_shape = spec_shape
        self.data_format = data_format
        self.frame_step = frame_step
        self.frame_length = frame_length

    def build(self, input_shape):
        self.mag_scale = self.add_weight(name='magnitude_scaling', 
                                         initializer=k.initializers.Constant(value=1.0),
                                         trainable=True)
        super(SimpleSpecLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return tf.TensorShape((None, self.spec_shape[0], self.spec_shape[1], 1))
        else:
            return tf.TensorShape((None, 1, self.spec_shape[0], self.spec_shape[1]))

    def call(self, inputs):

        # Perform STFT (complex64)
        complex_spec = tf.signal.stft(
            inputs,
            self.frame_length,
            self.frame_step,
            fft_length=self.frame_length,
            window_fn=tf.signal.hann_window,
            pad_end=False,
            name="stft"
        )

        # Compute power spectrum = real^2 + imag^2
        real = tf.math.real(complex_spec)
        imag = tf.math.imag(complex_spec)
        spec = real * real + imag * imag   # float32       

        # Fixed mag_scale: square root nonlinearity
        spec = tf.math.sqrt(spec)

        # Normalize values between 0 and 1 (corrected)
        min_val = k.backend.min(spec, axis=[1, 2], keepdims=True)
        max_val = k.backend.max(spec, axis=[1, 2], keepdims=True)
        spec = tf.math.divide(tf.math.subtract(spec, min_val), tf.math.maximum(max_val - min_val, 1e-8))

        # Print stats for debugging (only works in eager mode, but will show up in test_birdnet_simple.py)
        tf.print("SimpleSpecLayer spectrogram stats: min =", tf.reduce_min(spec), 
                 "max =", tf.reduce_max(spec), 
                 "mean =", tf.reduce_mean(spec), 
                 "std =", tf.math.reduce_std(spec))
        
        # Swap axes to fit input shape
        spec = tf.transpose(spec, [0, 2, 1])

        # Add channel axis        
        if self.data_format == 'channels_last':
            spec = tf.expand_dims(spec, -1)
        else:
            spec = tf.expand_dims(spec, 1)

        return spec

    def get_config(self):
        config = {'data_format': self.data_format,
                  'sample_rate': self.sample_rate,
                  'spec_shape': self.spec_shape,
                  'frame_step': self.frame_step,
                  'frame_length': self.frame_length}
        base_config = super(SimpleSpecLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
