import numpy as np
import h5py
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
from collections import Counter
from scipy.optimize import linear_sum_assignment


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
                    feature = feature_data[:-6]  # Extract the feature part
                    class_id = feature_data[-6]  # Extract the class_id
                    confidence = feature_data[-5]  # Extract the confidence score
                    bbox = feature_data[-4:-1]  # Extract the bounding box
                    overlap = feature_data[-1]  # Extract the IoU information
                    if isinstance(feature, np.ndarray) and feature.size > 0:  # Check if feature is valid
                        features[track_id].append((feature, class_id, confidence, bbox, overlap))
    return features


def separate_and_cluster_features(reid_features_dict, distance_threshold, confidence_threshold, iou_threshold):
    customer_features = []
    associate_features = []
    track_to_class = {}

    for track_id, features in reid_features_dict.items():
        track_key = track_id
        feature_list = [f[0] for f in features if isinstance(f[0], np.ndarray) and f[
            0].size > 0]  # Extract features only if they are valid numpy arrays
        class_id = features[0][1]  # Class ID is the same for all entries in a track
        confidences = [f[2] for f in features]
        overlaps = [f[4] for f in features]
        valid_indices = [i for i in range(len(confidences)) if
                         confidences[i] > confidence_threshold and not overlaps[i]]
        valid_features = [feature_list[i] for i in valid_indices]
        if class_id == 0:
            customer_features.extend(valid_features)
        else:
            associate_features.extend(valid_features)
        track_to_class[track_key] = class_id

    customer_features = normalize(np.array(customer_features), axis=1) if customer_features else np.array([])
    associate_features = normalize(np.array(associate_features), axis=1) if associate_features else np.array([])

    customer_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    customer_labels = customer_clustering.fit_predict(customer_features) if len(customer_features) > 0 else np.array([])

    associate_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    associate_labels = associate_clustering.fit_predict(associate_features) if len(
        associate_features) > 0 else np.array([])

    return customer_features, customer_labels, associate_features, associate_labels, track_to_class


def map_track_id_to_global_id(reid_features_dict, customer_features, customer_labels, associate_features,
                              associate_labels, track_to_class):
    unique_tracks = list(reid_features_dict.keys())
    track_to_global_id = {}

    global_customer_id_offset = 10000  # To ensure unique global IDs
    global_associate_id_offset = 20000

    for track in unique_tracks:
        track_key = track
        if track_key in reid_features_dict:
            class_id = track_to_class[track_key]
            track_features = np.array([f[0] for f in reid_features_dict[track_key] if
                                       isinstance(f[0], np.ndarray) and f[
                                           0].size > 0])  # Extract features only if they are valid numpy arrays
            confidences = np.array([f[2] for f in reid_features_dict[track_key]])  # Extract confidence scores only

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

            # Compute similarities and find the best match
            similarities = np.array([1 - cosine(best_feature, feature_list[i]) for i in range(len(label_list))])
            best_match = np.argmax(similarities)
            track_to_global_id[track] = label_list[best_match] + offset

    return track_to_global_id
