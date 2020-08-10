from tensorflow.keras.callbacks import Callback

class NeptuneLogger(Callback):
    def __init__(self, experiment):
      self.experiment = experiment

    def on_batch_end(self, batch, logs={}):
        for log_name, log_value in logs.items():
            self.experiment.log_metric(f'batch_{log_name}', log_value)

    def on_epoch_end(self, epoch, logs={}):
        for log_name, log_value in logs.items():
            self.experiment.log_metric(f'epoch_{log_name}', log_value)