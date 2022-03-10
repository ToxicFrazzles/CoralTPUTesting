import sys
from pathlib import Path
from model import Model
import datetime


def print_help():
    print()
    print("Classifcation test")
    print(f"Usage: {sys.argv[0].split('/')[-1]} [options]")
    print("  Options:")
    print("    -h, --help                    Shows this help text")
    print("    -t=<training-directory>       The directory to look for training data")
    print("    -w=<weights-path>             Path to either load weights or save weights to")
    print("                                      If a directory is specified, versions of the weights are saved")
    print("    -c=<classification-directory> Image files in this directory are classified into the correct sub-directory")
    print("    -e=<epoch-count>              Number of training epochs to perform before stopping")
    print()


class ConfigState:
    training_dir = None
    weights_path = None
    classification_dir = None
    epochs = None

    def __init__(self):
        try:
            for arg in sys.argv:
                # Don't handle help arg. It's handled before initialising this
                split_arg = arg.split("=")
                if split_arg[0] in ["-t"]:
                    self.training_dir = Path(split_arg[1])
                elif split_arg[0] in ["-w"]:
                    self.weights_path = Path(split_arg[1])
                elif split_arg[0] in ["-c"]:
                    self.classification_dir = Path(split_arg[1])
                elif split_arg[0] in ["-e"]:
                    self.epochs = int(split_arg[1])
        except Exception as e:
            print_help()
            print(e)
            print("Incorrect arguments. Please double-check they match the instructions provided above")
            exit(1)
        if self.weights_path is None:
            print("The weights path is required")
            exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1 or "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        exit(0)
    config = ConfigState()
    # if config.classification_dir is None:
    #     print("Nothing is implemented...")
    #     exit(0)

    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    tomorrow_morning = datetime.datetime.combine(tomorrow, datetime.time(6))
    this_evening = datetime.datetime.combine(datetime.date.today(), datetime.time(5, 45))

    model = Model(config.weights_path, config.training_dir)
    # while datetime.datetime.now() < this_evening:
    #     model.train(4)
    #     model.save_weights()
    model.set_quantization_aware(True)
    # for _ in range(5):
    #     model.train(4)
    #     model.save_tflite_8bit()
    model.train(1)
    model.save_tflite_8bit()
    r = model.predict(Path("sample_images/Photo_1.jpg"))
    print(r)
    prediction = max(r, key=r.get)
    confidence = r[prediction]
    print(f"Think it is: {prediction}")
    print(f"Confidence: {confidence}")
