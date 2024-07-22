import os
import argparse
import cv2
from detector_and_reid import DetectorAndReID
from utils import load_tracking_data, load_reid_features, separate_and_cluster_features, map_track_id_to_global_id


def make_parser():
    parser = argparse.ArgumentParser("Complete Video Processing with YOLOv8 and Clustering")
    parser.add_argument("--root_path", type=str, default="assets")
    parser.add_argument("-fps", "--sampling_rate", type=int, default=10,
                        help="Frames per second to sample for processing")
    parser.add_argument("--distance_threshold", type=float, default=1.3, help="Distance threshold for clustering")
    parser.add_argument("--iou_threshold", type=float, default=0.1,
                        help="IoU threshold for filtering overlapping detections")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="Confidence threshold for filtering detections")
    return parser


def process_video(video_path, detector_reid, fps, camera_id, scene_id, root_path):
    video_name = os.path.basename(video_path)
    emb_fol_path = os.path.join(root_path, "data/test_emb")
    det_fol_path = os.path.join(root_path, "data/test_det")
    embeddings_file_path = os.path.join(emb_fol_path, f"{scene_id}_emb.h5")
    detections_file_path = os.path.join(det_fol_path, f"{scene_id}_detections.txt")
    video = cv2.VideoCapture(video_path)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate / fps)
    frame_count = 0
    track_feature_count = {}

    with open(detections_file_path, 'w') as det_file:
        with h5py.File(embeddings_file_path, 'a') as hf:
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections, class_ids, confidences, track_ids, features, track_feature_count = detector_reid.detect_and_extract_features(
                    frame_rgb, track_feature_count, confidence_threshold=args.confidence_threshold,
                    iou_threshold=args.iou_threshold)

                # Save all detections to the .txt file
                for detection, track_id, class_id, score in zip(detections, track_ids, class_ids, confidences):
                    x1, y1, x2, y2 = detection
                    det_file.write(f'{frame_count},{track_id},{x1},{y1},{x2},{y2},{score},{camera_id},{class_id}\n')

                # Save only non-empty features to the .h5 file
                for feature, track_id, class_id, confidence, bbox, overlap in features:
                    if isinstance(feature, np.ndarray) and feature.size > 0:  # Check if feature is not empty
                        if f"track_{camera_id}_{track_id}" not in hf:
                            hf.create_group(f"track_{camera_id}_{track_id}")
                        data = np.hstack((feature, [class_id, confidence], bbox, overlap))
                        hf.create_dataset(
                            name=f"track_{camera_id}_{track_id}/idx_{len(hf[f'track_{camera_id}_{track_id}'])}",
                            data=data)

                frame_count += frame_interval

    video.release()


def main():
    args = make_parser().parse_args()
    cams = ['c001']
    scenes = ['S001']
    root_path = args.root_path
    reid1 = str(os.path.join(root_path, 'models/recent_finetuned.pth'))
    reid_model_paths = [('osnet_x1_0', reid1)]
    yolov8_weight_path = str(os.path.join(root_path, 'models/cso-8-class-yolov8-large-v1-640.pt'))
    det_reid = DetectorAndReID(
        yolo_weights_path=yolov8_weight_path,
        reid_model_paths=reid_model_paths,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    test_path = os.path.join(root_path, 'data/test/')
    for scene in scenes:
        scene_path = os.path.join(test_path, scene)
        for cam in cams:
            cam_path = os.path.join(scene_path, cam)
            if os.path.exists(cam_path):
                process_video(
                    video_path=os.path.join(cam_path, 'video.mp4'),
                    detector_reid=det_reid,
                    fps=args.sampling_rate, camera_id=cam, scene_id=scene, root_path=root_path)
            else:
                print("Video path is not found at", cam_path)

    # Perform clustering and generate global IDs
    tracking_data = load_tracking_data(os.path.join(root_path, 'data/test_det/S001_detections.txt'))
    reid_features_dict = load_reid_features(os.path.join(root_path, 'data/test_emb/S001_emb.h5'))
    customer_features, customer_labels, associate_features, associate_labels, track_to_class = separate_and_cluster_features(
        reid_features_dict, args.distance_threshold, args.confidence_threshold, args.iou_threshold)

    track_to_global_id = map_track_id_to_global_id(reid_features_dict, customer_features, customer_labels,
                                                   associate_features, associate_labels, track_to_class)

    # Map all detections to global IDs
    final_outputs = {cam: [] for cam in cams}
    for detection in tracking_data:
        frame_number, track_id, x1, y1, x2, y2, score, camera_id, class_id = detection
        global_id = track_to_global_id.get(f'track_{camera_id}_{track_id}', -1)
        final_outputs[camera_id].append([frame_number, global_id, x1, y1, x2, y2, score])

    for cam in cams:
        final_output = np.array(final_outputs[cam])
        output_dir = os.path.join(root_path, f"data/S001_output/{cam}")
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(os.path.join(output_dir, 'final_output.txt'), final_output, fmt='%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.2f')


if __name__ == '__main__':
    main()
