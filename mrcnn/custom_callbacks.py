import keras as k

class monitor_translation_during_training(k.callbacks.Callback):
    def __init__(self, x):
        self.loss_translations = None
        self.loss_rotations = None



    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        outputs = [self.model.get_layer("mrcnn_pose_loss/y_pred_t").output,
                   self.model.get_layer("mrcnn_pose_loss/y_pred_r").output]
        target_outputs = [self.model.get_layer("mrcnn_pose_loss/y_true_t").output,
                   self.model.get_layer("mrcnn_pose_loss/y_true_r").output]
        self.pred_func = k.backend.function(inputs=self.model.inputs, outputs=outputs)
        self.target_func = k.backend.function(inputs=self.model.inputs, outputs=target_outputs)
        self.loss_translations = []
        self.loss_rotations = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        predictions = self.model.predict_on_batch(self.x)
        loss_preds = self.pred_func(self.x)
        self.loss_translations.append(loss_preds[0])
        self.loss_rotations.append(loss_preds[1])
        return