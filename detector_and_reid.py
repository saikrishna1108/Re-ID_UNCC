import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import h5py
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from utils import iou

class DetectorAndReID:
    def __init__(self, yolo_weights_path, reid_model_paths, device='cpu'):
        self.device = torch.device(device)
        self.yolov8_model = YOLO(yolo_weights_path)
        self.extractors = [FeatureExtractor(model_name=model_name, model_path=model_path, device=device) for model_name, model_path in reid_model_paths]
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def detect_and_extract_features(self, img, track_feature_count):
        results = self.yolov8_model.track(img, persist=True)
        detections = results[0].boxes.xyxy.to('cpu').numpy()
        class_ids = results[0].boxes.cls.to('cpu').numpy()
        confidences = results[0].boxes.conf.to('cpu').numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

        # Calculate IoU for detections within the same frame
        ious = np.zeros((len(detections), len(detections)))
        for i in range(len(detections)):
            for j in range(len(detections)):
                if i != j:
                    ious[i, j] = iou(detections[i], detections[j])

        features, track_feature_count = self.get_features(img, detections, track_ids, class_ids, confidences, track_feature_count, ious)
        return detections, class_ids, confidences, track_ids, features, track_feature_count

    def get_features(self, inp_img, detections, track_ids, class_ids, confidences, track_feature_count, ious):
        inp_img = Image.fromarray(inp_img)
        detection_emb = []

        def generate_fibonacci_series(n):
            fib_series = [0, 1]
            while len(fib_series) < n:
                fib_series.append(fib_series[-1] + fib_series[-2])
            return fib_series[1:]  # Skip the first zero

        for idx, (x1, y1, x2, y2) in enumerate(detections):
            if idx < len(track_ids):
                track_id = track_ids[idx]
                class_id = class_ids[idx]
                confidence = confidences[idx]
                if track_id not in track_feature_count:
                    track_feature_count[track_id] = 0

                fib_index = track_feature_count[track_id]
                fib_series = generate_fibonacci_series(fib_index + 1)
                if fib_index < len(fib_series):
                    interval = fib_series[fib_index]
                    if (idx + 1) % interval == 0:
                        img_crop = inp_img.crop((x1, y1, x2, y2))
                        img_crop = self.transform(img_crop.convert('RGB')).unsqueeze(0)
                        features = [extractor(img_crop).cpu().detach().numpy()[0] for extractor in self.extractors]
                        feature = np.concatenate(features, axis=0)
                        if isinstance(feature, np.ndarray) and feature.size > 0:
                            detection_emb.append((feature, track_id, class_id, confidence, (x1, y1, x2, y2), ious[idx]))
                            track_feature_count[track_id] += 1
        return detection_emb, track_feature_count
