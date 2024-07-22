import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor


class DetectorAndReID:
    def __init__(self, yolo_weights_path, reid_model_paths, device='cpu'):
        self.device = torch.device(device)
        self.yolov8_model = YOLO(yolo_weights_path)
        self.extractors = [FeatureExtractor(model_name=model_name, model_path=model_path, device=device) for
                           model_name, model_path in reid_model_paths]
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def detect_and_extract_features(self, img, track_feature_count, confidence_threshold=0.5, iou_threshold=0.1):
        results = self.yolov8_model.track(img, persist=True)
        detections = results[0].boxes.xyxy.to('cpu').numpy()
        class_ids = results[0].boxes.cls.to('cpu').numpy()
        confidences = results[0].boxes.conf.to('cpu').numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

        valid_indices = [i for i in range(len(confidences)) if
                         confidences[i] > confidence_threshold and len(track_ids) > 0]
        detections = detections[valid_indices]
        class_ids = class_ids[valid_indices]
        confidences = confidences[valid_indices]
        track_ids = [track_ids[i] for i in valid_indices]

        overlaps = [False] * len(detections)
        for i in range(len(detections)):
            for j in range(len(detections)):
                if i != j and iou(detections[i], detections[j]) > iou_threshold:
                    overlaps[i] = True
                    break

        features = self.get_features(img, detections, track_ids, class_ids, track_feature_count, confidences, overlaps)
        return detections, class_ids, confidences, track_ids, features, track_feature_count

    def get_features(self, inp_img, detections, track_ids, class_ids, track_feature_count, confidences, overlaps):
        inp_img = Image.fromarray(inp_img)
        detection_emb = []

        def generate_fibonacci_series(n):
            fib_series = [0, 1]
            while len(fib_series) < n:
                fib_series.append(fib_series[-1] + fib_series[-2])
            return fib_series[1:]  # Skip the first zero

        for idx, (x1, y1, x2, y2) in enumerate(detections):
            track_id = track_ids[idx]
            class_id = class_ids[idx]
            confidence = confidences[idx]
            overlap = overlaps[idx]

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
                    detection_emb.append((feature, track_id, class_id, confidence, [x1, y1, x2, y2], overlap))
                    track_feature_count[track_id] += 1

        return detection_emb


def iou(box1, box2):
    x1_max, y1_max, x2_max, y2_max = box1
    x1_min, y1_min, x2_min, y2_min = box2

    xi1 = max(x1_max, x1_min)
    yi1 = max(y1_max, y1_min)
    xi2 = min(x2_max, x2_min)
    yi2 = min(y2_max, y2_min)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2_max - x1_max) * (y2_max - y1_max)
    box2_area = (x2_min - x1_min) * (y2_min - y1_min)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou
