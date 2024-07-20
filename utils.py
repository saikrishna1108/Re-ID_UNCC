import numpy as np
import h5py
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
from collections import Counter
from scipy.optimize import linear_sum_assignment

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

def load_tracking_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            data.append([float(p) if i != 7 else p for i, p in enumerate(parts)])
    return np.array(data, dtype=object)

def load_reid_features(file_path):
    features = {}
    with h5py.File(file_path, 'r') as f:
        for track_id in f.keys():
            features[track_id] = []
            for idx in f[track_id].keys():
                dataset = f[f"{track_id}/{idx}"]
                if isinstance(dataset, h5py.Dataset) and dataset.shape != ():  # Ensure it is not a scalar
                    feature_data = dataset[:]
                    feature = feature_data[:-7]  # Extract the feature part
                    class_id = feature_data[-7]  # Extract the class_id
                    confidence = feature_data[-6]  # Extract the confidence score
                    bbox = feature_data[-5:-1]  # Extract the bounding box
                    ious = feature_data[-1]  # Extract the IoU information
                    if isinstance(feature, np.ndarray) and feature.size > 0:  # Check if feature is valid
                        features[track_id].append((feature, class_id, confidence, bbox, ious))
    return features

def separate_and_cluster_features(reid_features_dict, distance_threshold, confidence_threshold, iou_threshold):
    customer_features = []
    associate_features = []
    customer_confidences = []
    associate_confidences = []
    customer_bboxes = []
    associate_bboxes = []
    customer_ious = []
    associate_ious = []
    track_to_class = {}
    
    for track_id, features in reid_features_dict.items():
        track_key = track_id
        feature_list = [f[0] for f in features if isinstance(f[0], np.ndarray) and f[0].size > 0]  # Extract features only if they are valid numpy arrays
        class_id = features[0][1]  # Class ID is the same for all entries in a track
        confidence_list = [f[2] for f in features]  # Extract confidence scores only
        bbox_list = [f[3] for f in features]  # Extract bounding boxes only
        iou_list = [f[4] for f in features]  # Extract IoU information only
        if class_id == 0:
            customer_features.extend(feature_list)
            customer_confidences.extend(confidence_list)
            customer_bboxes.extend(bbox_list)
            customer_ious.extend(iou_list)
        else:
            associate_features.extend(feature_list)
            associate_confidences.extend(confidence_list)
            associate_bboxes.extend(bbox_list)
            associate_ious.extend(iou_list)
        track_to_class[track_key] = class_id

    # Ensure all features are valid numpy arrays
    customer_features = [f for f in customer_features if isinstance(f, np.ndarray) and f.size > 0]
    associate_features = [f for f in associate_features if isinstance(f, np.ndarray) and f.size > 0]

    # Normalize the features
    if len(customer_features) > 0:
        customer_features = np.array(customer_features)
        customer_features = normalize(customer_features, axis=1)
    if len(associate_features) > 0:
        associate_features = np.array(associate_features)
        associate_features = normalize(associate_features, axis=1)
    
    # Filter features based on confidence score and overlap
    def filter_features(features, confidences, bboxes, ious, confidence_threshold, iou_threshold):
        valid_indices = [i for i in range(len(confidences)) if confidences[i] > confidence_threshold]
        filtered_features = np.array([features[i] for i in valid_indices])
        filtered_bboxes = np.array([bboxes[i] for i in valid_indices])
        filtered_ious = np.array([ious[i] for i in valid_indices])
        
        non_overlapping_indices = []
        for i in range(len(filtered_features)):
            if np.all(filtered_ious[i] <= iou_threshold):
                non_overlapping_indices.append(i)
        
        filtered_features = filtered_features[non_overlapping_indices]
        filtered_bboxes = filtered_bboxes[non_overlapping_indices]
        return filtered_features, filtered_bboxes

    customer_features, customer_bboxes = filter_features(customer_features, customer_confidences, customer_bboxes, customer_ious, confidence_threshold, iou_threshold)
    associate_features, associate_bboxes = filter_features(associate_features, associate_confidences, associate_bboxes, associate_ious, confidence_threshold, iou_threshold)
    
    # Clustering
    if len(customer_features) > 0:
        customer_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
        customer_labels = customer_clustering.fit_predict(customer_features)
        print("Number of customers ", max(customer_labels) + 1)
    else:
        customer_labels = np.array([])

    if len(associate_features) > 0:
        associate_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
        associate_labels = associate_clustering.fit_predict(associate_features)
        print("Number of associates ", max(associate_labels) + 1)
    else:
        associate_labels = np.array([])

    return customer_features, customer_labels, associate_features, associate_labels, track_to_class

def map_track_id_to_global_id(reid_features_dict, customer_features, customer_labels, associate_features, associate_labels, track_to_class):
    unique_tracks = list(reid_features_dict.keys())
    track_to_global_id = {}
    
    global_customer_id_offset = 10000  # To ensure unique global IDs
    global_associate_id_offset = 20000

    for track in unique_tracks:
        track_key = track
        if track_key in reid_features_dict:
            class_id = track_to_class[track_key]
            track_features = np.array([f[0] for f in reid_features_dict[track_key] if isinstance(f[0], np.ndarray) and f[0].size > 0])  # Extract features only if they are valid numpy arrays
            confidences = np.array([f[2] for f in reid_features_dict[track_key]])  # Extract confidence scores only
            
            # Find the feature with the highest confidence score
            best_feature_idx = np.argmax(confidences)
            best_feature = track_features[best_feature_idx]
            
            if class_id == 0:  # Customer
                label_list = customer_labels
                feature_list = customer_features
                offset = global_customer_id_offset
            else:  # Associate
                label_list = associate_labels
                feature_list = associate_features
                offset = global_associate_id_offset
            
            # Normalize the best feature
            best_feature = normalize(best_feature.reshape(1, -1), axis=1)
            
            # Compute similarities and find the best match
            similarities = np.array([1 - cosine(best_feature, feature_list[i].reshape(1, -1)) for i in range(len(label_list))])
            best_match = np.argmax(similarities)
            track_to_global_id[track] = label_list[best_match] + offset

    return track_to_global_id
