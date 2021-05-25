 class callbacks(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
            if (logs.get('acc')>0.99):
                print('Reached desired accuracy cancelling Training!')
                self.model.stop_training=True

    callback=callbacks()
