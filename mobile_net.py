import threading
import time

import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class MobileNet:
    def __init__(self, weights_path, device, threshold):
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn()
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        self.model.to(device)

        self.threshold = threshold

        self.lock = threading.Lock()
        self.thread = None

        self.result = None
        self.finished = True

    def run(self, frame):
        temp_result = self.model(frame)[0]
        if len(temp_result['scores']) > 0:
            scores = temp_result['scores'].to("cpu").detach().numpy()
            max_id = np.argmax(scores)
            max_score = scores[max_id]
            if max_score > self.threshold:
                temp_result = temp_result['boxes'].to("cpu").detach().numpy().astype(int)[max_id]
            else:
                temp_result = None
        else:
            temp_result = None
        with self.lock:
            self.result = temp_result
            self.finished = True

    def run_async(self, frame):
        self.finished = False
        self.thread = threading.Thread(target=self.run, args=(frame,))
        self.thread.start()

    def is_finished(self):
        return self.finished

    def get(self):
        if self.is_finished():
            if self.thread is not None:
                self.thread.join()
            return self.result
        else:
            return None
