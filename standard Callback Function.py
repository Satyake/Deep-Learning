 class callbacks(tf.keras.callbacks.Callback):
        def epoch_end(self,epoch,logs={}):
            if (logs.get('accuracy')>0.99):
                print('Reached desired accuracy cancelling Training!')
                self.model.stop_training=True
