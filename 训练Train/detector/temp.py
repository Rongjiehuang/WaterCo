from model import MaskRCNN
from config import Config
import os
ROOT_DIR = os.path.abspath("./models")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
class TacoTestConfig(Config):
    NAME = "taco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0
    NUM_CLASSES = 10
    USE_OBJECT_ZOOM = False

config = TacoTestConfig()



model = MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)
model.summary()