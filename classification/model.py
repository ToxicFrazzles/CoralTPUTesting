# We will use EfficientNetV2L as it should fit on the EdgeTPU and result in a reasonable accuracy with a low-enough latency
from pathlib import Path
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tflite_runtime.interpreter
from keras.applications.efficientnet_v2 import EfficientNetV2L
from keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
from PIL import Image
import json
import math
from typing import List
from random import shuffle


BATCH_SIZE = 16


def preprocess_img(img_path, batch=True):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((480, 480))
    img_arr = np.asarray(img)
    if batch:
        img_arr = img_arr[None, :, :, :3]
    img_arr = 2.0 * (img_arr - img_arr.min()) / img_arr.ptp() -1
    return img_arr.astype('float32')


class TrainingSequence(tf.keras.utils.Sequence):
    def __init__(self, training_dir: Path, labels: List[str], batch_size=BATCH_SIZE) -> None:
        self.training_dir: Path = training_dir
        self.labels: List[str] = labels
        self.batch_size = batch_size
        self.x: List[Path] = []
        self.y = []
        x = []
        y = []
        for i in range(len(self.labels)):
            label = self.labels[i]
            ldir = self.training_dir / label
            for file in ldir.iterdir():
                if not file.is_file():
                    continue
                if file.name == "1a391be.jpg":
                    print(self.labels)
                    print(i)
                x.append(file)
                y.append(i)
        indices = list(range(len(x)))
        shuffle(indices)
        for index in indices:
            self.x.append(x[index])
            self.y.append(y[index])
    
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]
        return np.array([
            preprocess_img(filename, batch=False)
            for filename in batch_x
        ]), np.array(batch_y)


class Model:
    def __init__(self, weights_path: Path, training_dir: Path = None):
        if not weights_path.is_dir():
            raise ValueError("The provided weights path is not a directory")
        self.weights_path = weights_path
        self.training_dir = training_dir
        self._load_labels()
        self.model = None
        self._load_model()
        self._qa_training = False

    def _load_model(self):
        self.model = MobileNetV2(
            include_top=True, weights=None, classes=len(self.labels),
            # include_preprocessing=False,
            input_shape=(480, 480, 3)
        )
        self.current_epoch = 0
        epoch, latest_weights_file = self._latest_weights()
        if latest_weights_file is not None:
            self.current_epoch = epoch
            self.model.load_weights(latest_weights_file)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    def _latest_weights(self):
        latest_epoch = None
        for file in self.weights_path.iterdir():
            if not file.is_file():
                continue
            if not file.suffix == ".h5":
                continue
            if not file.stem.startswith("epoch"):
                continue
            if latest_epoch is None or latest_epoch < int(file.stem[5:]):
                latest_epoch = int(file.stem[5:])
        if latest_epoch is None:
            return None, None
        return latest_epoch, self.weights_path / f"epoch{latest_epoch}.h5"

    def _load_labels(self):
        if self.training_dir is not None:
            self._labels_from_training_dir()
            return
        if not self._labels_from_weights_dir():
            raise ValueError("No training dir provided and no labels file exists in the weights directory")

    def _labels_from_weights_dir(self):
        if not self.weights_path.is_dir():
            return False
        if not (self.weights_path / "labels.json").is_file():
            return False
        try:
            with (self.weights_path / "labels.json").open() as f:
                self.labels = json.load(f)
            return True
        except json.JSONDecodeError:
            return False

    def _labels_from_training_dir(self):
        self.labels = [en.name for en in self.training_dir.iterdir() if en.is_dir()]
        if self.weights_path is not None and self.weights_path.is_dir():
            with (self.weights_path / "labels.json").open("w") as f:
                json.dump(self.labels, f)
        return True

    def predict(self, image_path):
        img_arr = preprocess_img(image_path)
        r = self.model.predict(img_arr)[0]
        return {self.labels[i]: res for i, res in enumerate(r)}
    
    def save_weights(self):
        filename = f"{'QA' if self._qa_training else ''}epoch{self.current_epoch}.h5"
        self.model.save_weights(self.weights_path / filename)

    def train(self, epochs):
        if self.model.compiled_loss is None:
            print("Model not compiled...")
            return
        self.model.fit(
            x=TrainingSequence(self.training_dir, self.labels),
            batch_size=BATCH_SIZE, epochs=epochs
        )
        self.current_epoch += epochs

    def save_tflite_8bit(self):
        if not self._qa_training:
            print("Cannot save model to TFLite that is not quantization aware")
            print("First use the `set_quantization_aware(True)` method and try again")
            return False
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.int8
        # converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        with (self.weights_path / f"8bit_epoch{self.current_epoch}.tflite").open('wb') as f:
            f.write(tflite_model)

    def set_quantization_aware(self, value: bool):
        if self._qa_training and value:
            return
        elif not self._qa_training and not value:
            return
        elif value:
            self._qa_training = True
            self.model = tfmot.quantization.keras.quantize_model(self.model)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            )
        else:
            self._qa_training = False
            self.model = None
            tf.keras.backend.clear_session()
            self._load_model()
