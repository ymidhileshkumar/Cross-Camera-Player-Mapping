import os
import random
import cv2
import numpy as np
from ultralytics import YOLO
import json
from collections import defaultdict
from video_processor import (
    MultiMatchAnalyzer, MatchAnalyzer, run_matching_pipeline, ColorClusterAnalyzer
)
from video_processor import VideoProcessor as VPro
# Deep SORT imports (adjust these imports if your directory structure is different)
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import argparse
class Track:
    def __init__(self, track_id, bbox,class_id=None):
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        

class DeepSortYoloTracker:
    def __init__(self, encoder_model_filename, max_cosine_distance=0.4, nn_budget=None):
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)
        self.tracks = []
        self.track_id_to_class_id = {}

    def update(self, frame, detections):
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self._update_tracks()
            return

        bboxes = np.asarray([d[:-2] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-2] for d in detections]
        class_ids = [d[-1] for d in detections]
        features = self.encoder(frame, bboxes)

        dets = [Detection(bbox, scores[i], features[i]) for i, bbox in enumerate(bboxes)]
        self.tracker.predict()
        self.tracker.update(dets)
         # Update mapping from detection to class_id
        for i, track in enumerate(self.tracker.tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if i < len(class_ids):
                self.track_id_to_class_id[track.track_id] = class_ids[i]
        self._update_tracks()

    def _update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            track_id = track.track_id
            class_id=self.track_id_to_class_id.get(track_id,None)
            # tracks.append(Track(track_id, bbox))
            tracks.append(Track(track_id, bbox, class_id))
        self.tracks = tracks

class VideoProcessor:
    def __init__(self, input_video_path, output_video_path, yolo_model_path, encoder_model_path,crops_root, detection_threshold=0.5):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.model = YOLO(yolo_model_path)
        self.tracker = DeepSortYoloTracker(encoder_model_path)
        self.detection_threshold = detection_threshold
        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]
        self.class_names = self.model.names 
        self.tracking_data=defaultdict(list)
        self.crops_root=crops_root

    def process(self):
        cap = cv2.VideoCapture(self.input_video_path)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read video: {self.input_video_path}")
            return

    
        
        os.makedirs(self.crops_root, exist_ok=True)
        
        frame_index=0

        while ret:
            results = self.model(frame)
            ret, frame = cap.read() 
            if not ret or frame is None:
                print("Failed to read frame")
                continue  # or break, depending on your logic
            for result in results:
                detections = []
                for r in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    if score > self.detection_threshold:
                        detections.append([int(x1), int(y1), int(x2), int(y2), score,int(class_id)])
                self.tracker.update(frame, detections)
                frame_tracks = []
                for track in self.tracker.tracks:
                    x1, y1, x2, y2 = map(int, track.bbox)
                    frame_tracks.append({
                        "track_id": track.track_id,
                        "bbox": [x1, y1, x2, y2],
                        "class_id": track.class_id,
                        "class_name": self.class_names.get(track.class_id, "Unknown")
                    })
                    h, w = frame.shape[:2]
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(w, x2), min(h, y2)
                    crop_img = frame[y1c:y2c, x1c:x2c]

                    # Make folder for this track ID if it doesn't exist
                    id_folder = os.path.join(self.crops_root, f"id_{track.track_id}")
                    os.makedirs(id_folder, exist_ok=True)

                    # Save crop with desired naming
                    crop_filename = f"frame_{frame_index}_id{track.track_id}.png"
                    crop_path = os.path.join(id_folder, crop_filename)
                    if crop_img.size > 0:
                        cv2.imwrite(crop_path, crop_img)
                       
            
                self.tracking_data[frame_index] = frame_tracks
                frame_index += 1
        
        # Save tracking data to JSON
        self._save_tracking_data()

    def _save_tracking_data(self):
        base_name = os.path.splitext(self.output_video_path)[0]
        json_path = f"{base_name}_tracking.json"
        
        # Convert defaultdict to regular dict for JSON serialization
        with open(json_path, 'w') as f:
            json.dump(dict(self.tracking_data), f, indent=4)
        print(f"Tracking data saved to {json_path}")    

def parse_args():
    parser = argparse.ArgumentParser(description="Custom tracking pipeline")
    parser.add_argument('--video1_path', type=str, required=True, help='Path to first video')
    parser.add_argument('--video2_path', type=str, required=True, help='Path to second video')
    parser.add_argument('--yolo_model_path', type=str, required=True, help='Path to YOLO model (.pt file)')
    # You can add more arguments as needed
    return parser.parse_args()


if __name__ == "__main__":
         
  

  
      args = parse_args()

      csv_path = "final_results_1.csv"
      final_output_path = "side_by_side_out.mp4"

      video_path_1 = args.video1_path
      video_out_path_1 = os.path.splitext(video_path_1)[0] + "_out.mp4"
      base_name_1 = os.path.splitext(video_out_path_1)[0]
      json_path_1 = f"{base_name_1}_tracking.json"
      yolo_model_path = args.yolo_model_path
      encoder_model_path = 'mars-small128.pb'
      crops_root_1 = os.path.splitext(video_path_1)[0] +"_out"

      video_path_2 = args.video2_path
      video_out_path_2 = os.path.splitext(video_path_2)[0] + "_out.mp4"
      base_name_2 = os.path.splitext(video_out_path_2)[0]
      json_path_2 = f"{base_name_2}_tracking.json"
      crops_root_2 = os.path.splitext(video_path_2)[0] +"_out"

    # ...rest of your code follows, unchanged...

    
      ###########tracking
      print("intializing tracking")

      processor_1 = VideoProcessor(
          input_video_path=video_path_1,
          output_video_path=video_out_path_1,
          yolo_model_path=yolo_model_path,
          encoder_model_path=encoder_model_path,
          crops_root=crops_root_1,
          detection_threshold=0.5
          
      )
      processor_1.process()
      
      

     

      processor_2 = VideoProcessor(
          input_video_path=video_path_2,
          output_video_path=video_out_path_2,
          yolo_model_path=yolo_model_path,
          encoder_model_path=encoder_model_path,
          crops_root=crops_root_2,
          detection_threshold=0.5
      )
      processor_2.process()
      

   ###########################clustering
      print("intializing clustering")
      analyzer = ColorClusterAnalyzer(crops_root_1, crops_root_2, n_clusters=4)
      analyzer.run()
      csv_path=csv_path
      results=analyzer.save_and_return_results(csv_path)
      analyzer.plot_clusters("cluster_plot_1.png")
      MATCHING_CONFIG = {
            'csv_path': csv_path,
            'broadcast_base': crops_root_1,
            'tactimian_base': crops_root_2,
            'min_match': 2,
            'max_workers': 6
        }

      detailed_results = run_matching_pipeline(MATCHING_CONFIG)
      multi_analyzer = MultiMatchAnalyzer(detailed_results).run_analysis()
      combined_best_matches = multi_analyzer.combined_best_matches

      VIDEO_CONFIG = {
            "json1_path": json_path_1 ,
            "json2_path": json_path_2,
            "video1_path": video_path_1,
            "video2_path": video_path_2,
            "output_path":final_output_path
        }

      processor = VPro(
            video1_path=VIDEO_CONFIG["video1_path"],
            video2_path=VIDEO_CONFIG["video2_path"],
            json1_path=VIDEO_CONFIG["json1_path"],
            json2_path=VIDEO_CONFIG["json2_path"],
            output_path=VIDEO_CONFIG["output_path"],
            best_matches=combined_best_matches,
            single_output_path1=video_out_path_1,
            single_output_path2=video_out_path_2,
            draw_lines=False
        )
      processor.process_videos()
    
      
      
