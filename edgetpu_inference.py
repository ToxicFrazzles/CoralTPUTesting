import operator
import tflite_runtime.interpreter as tflite
from pathlib import Path
import platform
import json
from PIL import Image
import sys
import numpy as np
import collections
import time


WEIGHTS_DIR = Path(__file__).parent / "weights"
EDGETPU_SHARED_LIBRARY = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


Class = collections.namedtuple('Class', ['id', 'score'])


def preprocess_img(img_path, batch=True):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((480, 480))
    img_arr = np.asarray(img)/255
    if batch:
        img_arr = img_arr[None, :, :, :3]
    # img_arr = 2.0 * (img_arr - img_arr.min()) / img_arr.ptp() -1
    return img_arr.astype('float32')


class EdgeTPUClassifier:
    def __init__(self, weights_dir: Path):
        self._weights_dir = weights_dir
        self.labels = self.load_labels()
        self.interpreter = self.make_interpreter()
        self.interpreter.allocate_tensors()

    def get_latest_weights(self) -> Path:
        latest_epoch = None
        for entity in self._weights_dir.iterdir():
            if not entity.is_file() or entity.suffix != '.tflite':
                continue
            filename = entity.stem
            if filename.endswith("_edgetpu") and filename.startswith("8bit_epoch"):
                epoch = int(filename.split("_")[1][5:])
                if latest_epoch is None or epoch > latest_epoch:
                    latest_epoch = epoch
        return self._weights_dir / f"8bit_epoch{latest_epoch}_edgetpu.tflite"

    def load_labels(self):
        if not (self._weights_dir / "labels.json").is_file():
            raise Exception("No labels file found in the weights directory")
        with (self._weights_dir / "labels.json").open() as f:
            return json.load(f)

    def make_interpreter(self) -> tflite.Interpreter:
        return tflite.Interpreter(
            model_path=str(self.get_latest_weights()),
            experimental_delegates=[
                tflite.load_delegate(
                    EDGETPU_SHARED_LIBRARY,
                    {}
                )
            ]
        )

    def input_details(self, key):
        return self.interpreter.get_input_details()[0][key]

    def input_size(self):
        _, height, width, _ = self.input_details('shape')
        return width, height

    def input_tensor(self):
        tensor_index = self.input_details('index')
        return self.interpreter.tensor(tensor_index)()[0]

    def output_tensor(self, dequantize=True):
        output_details = self.interpreter.get_output_details()[0]
        output_data = np.squeeze(self.interpreter.tensor(output_details['index'])())
        if dequantize and np.issubdtype(output_details['dtype'], np.integer):
            scale, zero_point = output_details['quantization']
            return scale * (output_data - zero_point)
        return output_data

    def set_input(self, data):
        self.input_tensor()[:, :] = data

    def get_output(self, top_k=1, score_threshold=0.0):
        scores = self.output_tensor()
        classes = [
            Class(i, scores[i])
            for i in np.argpartition(scores, -top_k)[-top_k:]
            if scores[i] >= score_threshold
        ]
        return sorted(classes, key=operator.itemgetter(1), reverse=True)

    def infer(self, image, top_k=1, score_threshold=0.0):
        self.set_input(image)
        self.interpreter.invoke()
        return self.get_output(top_k, score_threshold)


if __name__ == "__main__":
    classifier = EdgeTPUClassifier(WEIGHTS_DIR)
    image = preprocess_img(Path(sys.argv[1]), False)
    result = classifier.infer(image, 3)

    benchmark_iterations = 50

    start = time.perf_counter()
    for _ in range(benchmark_iterations):
        classifier.infer(image)
    end = time.perf_counter()
    duration = end - start
    print(f"Took {duration * 1000:.5f} ms to run inference {benchmark_iterations} times")
    print(f"Averaged {duration * 1000 / benchmark_iterations:.5f} ms per inference")
    print(f"Or {benchmark_iterations / duration:.5f} inferences per second")

    print("----- RESULTS -----")
    for klass in result:
        print(f"{classifier.labels[klass.id]}: {klass.score:.5f}")
