import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

class TransferCNN:
    def __init__(self, input_shape=(128,128,3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False,
                                                input_shape=self.input_shape, pooling='avg')
        # Freeze base
        base.trainable = False

        inp = layers.Input(shape=self.input_shape)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
        x = base(x, training=False)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(self.num_classes, activation='softmax')(x)

        model = models.Model(inputs=inp, outputs=out)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def summary(self):
        self.model.summary()

    def train(self, train_ds, val_ds, epochs=10, out_path="models/cnn_transfer.h5"):
        cb = [
            callbacks.ModelCheckpoint(out_path, save_best_only=True, monitor="val_accuracy"),
            callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)
        ]
        history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cb)
        return history

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
