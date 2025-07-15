
import pandas as pd
import numpy as np
import cv2
import json
import os
from typing import Dict
from image_matching import run_matching_pipeline
from color_cluster import ColorClusterAnalyzer



class MultiMatchAnalyzer:
    def __init__(self, detailed_results: Dict[int, pd.DataFrame], weights=None):
        self.detailed_results = detailed_results
        self.weights = weights or {
            'total_matches': 0.05,
            'avg_match': 0.9,
            'max_match': 0.05
        }
        self.all_agg_dfs = []
        self.combined_best_matches = None
        self.analyzers = []

    def run_analysis(self):
        """Run analysis on all input data and combine results"""
        for cluster, df in self.detailed_results.items():
            analyzer = MatchAnalyzer(df=df, weights=self.weights).run_analysis()
            self.analyzers.append(analyzer)
            self.all_agg_dfs.append(analyzer.agg_df)

        # Combine results from all analyzers
        combined_agg = pd.concat(self.all_agg_dfs)
        self.combined_best_matches = combined_agg.sort_values(
            'match_score', ascending=False
        ).drop_duplicates('source_id')[['source_id', 'target_id', 'match_score']]
        
        return self

class MatchAnalyzer:
    def __init__(self, df: pd.DataFrame, weights=None):
        self.df = df
        self.agg_df = None
        self.match_matrix = None
        self.best_matches = None
        self.weights = weights or {
            'total_matches': 0.05,
            'avg_match': 0.9,
            'max_match': 0.05
        }
        self.metrics = ['total_matches', 'avg_match', 'max_match']

    def calculate_aggregates(self):
        self.agg_df = self.df.groupby(['source_id', 'target_id']).agg(
            total_matches=('match_count', 'count'),
            avg_match=('match_count', 'mean'),
            max_match=('match_count', 'max'),
            match_consistency=('match_count', lambda x: x.std())
        ).reset_index()
        return self
    
    
    def normalize_metrics(self):
       
        def safe_normalize(x):
            denom = x.max() - x.min()
            if denom == 0:
                return 1.0  # or 0.0, depending on your logic
            else:
                return (x - x.min()) / denom
    
        self.agg_df[self.metrics] = self.agg_df[self.metrics].apply(safe_normalize)

        return self

    def calculate_match_score(self):
        self.agg_df['match_score'] = sum(
            self.agg_df[metric] * self.weights[metric]
            for metric in self.metrics
        )
        return self

    def create_match_matrix(self):
        score_matrix = self.agg_df.pivot_table(
            index='source_id',
            columns='target_id',
            values='match_score',
            fill_value=0
        )
        self.match_matrix = score_matrix
        return self

    def find_best_matches(self):
        idx = self.agg_df.groupby('source_id')['match_score'].idxmax().dropna()
        self.best_matches = self.agg_df.loc[idx][['source_id', 'target_id', 'match_score']]
        # self.best_matches = self.agg_df.loc[
        #     self.agg_df.groupby('source_id')['match_score'].idxmax()
        # ][['source_id', 'target_id', 'match_score']]
        return self

    def run_analysis(self):
        return (self.calculate_aggregates()
                .normalize_metrics()
                .calculate_match_score()
                .create_match_matrix()
                .find_best_matches())

class VideoProcessor:
    DISTINCT_COLORS = [
        (255, 0, 0),    # Red for ID 00 (unmatched)
        (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 0, 255), (0, 255, 128),
        (128, 255, 0), (255, 0, 128), (0, 128, 255),
        (128, 0, 128), (0, 128, 128), (128, 128, 0)
    ]
    
    def __init__(self, video1_path, video2_path, json1_path, json2_path, output_path, best_matches, single_output_path1,
                 single_output_path2,
                 codec='mp4v',draw_lines=True):
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.json1_path = json1_path
        self.json2_path = json2_path
        self.output_path = output_path
        self.best_matches = best_matches
        self.cap1 = None
        self.cap2 = None
        self.out = None
        self.tracks1 = None
        self.tracks2 = None
        self.frame_width = 0
        self.frame_height = 0
        self.draw_lines = draw_lines
        self.output_path1 = single_output_path1
        self.output_path2 = single_output_path2

        # codec choice (mp4v, H264, etc.)
        self.codec = codec

        

    @staticmethod
    def get_color(track_id):
        return VideoProcessor.DISTINCT_COLORS[abs(track_id) % len(VideoProcessor.DISTINCT_COLORS)]

    def update_one_video(self, json_path, mapping, next_id_start):
        with open(json_path, 'r') as f:
            data = json.load(f)

        all_tracks = set()
        for frame_data in data.values():
            for track in frame_data:
                all_tracks.add(track['track_id'])

        unmatched = all_tracks - set(mapping.keys())
        for track_id in unmatched:
            mapping[track_id] = 0

        for frame_data in data.values():
            for track in frame_data:
                if track['track_id'] in mapping:
                    track['track_id'] = mapping[track['track_id']]

        new_path = json_path.replace('.json', '_updated.json')
        with open(new_path, 'w') as f:
            json.dump(data, f, indent=4)

        return next_id_start

    def update_global_ids(self):
        mapping_video1 = {}
        mapping_video2 = {}
        common_id = 1

        if not self.best_matches.empty:
            for _, row in self.best_matches.iterrows():
                try:
                    src_id = int(row['source_id'].split('_')[1])
                    tgt_id = int(row['target_id'].split('_')[1])
                    mapping_video1[src_id] = common_id
                    mapping_video2[tgt_id] = common_id
                    common_id += 1
                except (IndexError, ValueError):
                    continue

        next_id = self.update_one_video(self.json1_path, mapping_video1, common_id)
        self.update_one_video(self.json2_path, mapping_video2, next_id)

        return (self.json1_path.replace('.json', '_updated.json'),
                self.json2_path.replace('.json', '_updated.json'))

    def setup_video_capture(self):
        self.cap1 = cv2.VideoCapture(self.video1_path)
        self.cap2 = cv2.VideoCapture(self.video2_path)

        if not self.cap1.isOpened() or not self.cap2.isOpened():
            raise RuntimeError("Error opening video files!")

        self.frame_width = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap1.get(cv2.CAP_PROP_FPS)

        total_frames1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_offset = total_frames1 - total_frames2

        if frame_offset > 0:
            self.cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)
        elif frame_offset < 0:
            self.cap2.set(cv2.CAP_PROP_POS_FRAMES, -frame_offset)

        return fps

    def setup_video_writer(self, fps):
        self.out = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps if not np.isnan(fps) and fps > 0 else 30.0,
            (self.frame_width * 2, self.frame_height)
        )

    def load_tracking_data(self, json1_path, json2_path):
        with open(json1_path) as f1, open(json2_path) as f2:
            self.tracks1 = json.load(f1)
            self.tracks2 = json.load(f2)

    def process_frame(self, frame1, frame2, current_frame1, current_frame2):
        centers1, centers2, color_map = {}, {}, {}

        frame1_data = self.tracks1.get(str(current_frame1), [])
        for track in frame1_data:
            x1, y1, x2, y2 = map(int, track["bbox"])
            track_id = track['track_id']
            color = self.get_color(track_id)
            color_map[track_id] = color

            cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)
            display_id = f"{track_id:02d}" if track_id == 0 else str(track_id)
            cv2.putText(frame1, f"ID:{display_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            centers1[track_id] = ((x1 + x2) // 2, (y1 + y2) // 2)

        frame2_data = self.tracks2.get(str(current_frame2), [])
        for track in frame2_data:
            x1, y1, x2, y2 = map(int, track["bbox"])
            track_id = track['track_id']
            color = self.get_color(track_id)
            color_map[track_id] = color

            cv2.rectangle(frame2, (x1, y1), (x2, y2), color, 2)
            display_id = f"{track_id:02d}" if track_id == 0 else str(track_id)
            cv2.putText(frame2, f"ID:{display_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            centers2[track_id] = ((x1 + x2) // 2, (y1 + y2) // 2)

        combined = np.hstack((frame1, frame2))
        lines_drawn = 0
        
        if self.draw_lines:
            lines_drawn = self.draw_mapping_lines(combined, centers1, centers2, color_map)

        cv2.putText(combined, f"Video1 Frame: {current_frame1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, f"Video2 Frame: {current_frame2}",
                    (self.frame_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        cv2.putText(frame1, f"Video1 Frame: {current_frame1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame2, f"Video2 Frame: {current_frame2}",
                    (self.frame_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        return combined, lines_drawn,frame1,frame2

    def draw_mapping_lines(self, combined, centers1, centers2, color_map):
        lines_drawn = 0
        common_ids = set(centers1.keys()) & set(centers2.keys())

        for track_id in common_ids:
            if track_id != 0:
                pt1 = centers1[track_id]
                pt2 = (centers2[track_id][0] + self.frame_width, centers2[track_id][1])
                color = color_map[track_id]
                cv2.line(combined, pt1, pt2, color, 2)
                lines_drawn += 1

        return lines_drawn
    
   

    def process_videos(self):
           
        try:
            updated_json1, updated_json2 = self.update_global_ids()
            fps = self.setup_video_capture()
            self.setup_video_writer(fps)
            self.load_tracking_data(updated_json1, updated_json2)

            out1 = None
            out2 = None

            frame_idx = 0
            total_lines_drawn = 0

            while True:
                ret1, frame1 = self.cap1.read()
                ret2, frame2 = self.cap2.read()
                if not ret1 or not ret2:
                    break

                current_frame1 = int(self.cap1.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                current_frame2 = int(self.cap2.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                combined_frame, lines_drawn ,_,_= self.process_frame(
                    frame1, frame2, current_frame1, current_frame2
                )

                # write combined
                self.out.write(combined_frame)

                # lazy-init per-camera writers
                if frame_idx == 0:
                    h, w = combined_frame.shape[:2]
                    half_w = w // 2
                    size = (half_w, h)
                    fourcc = cv2.VideoWriter_fourcc(*self.codec)
                    out1 = cv2.VideoWriter(self.output_path1, fourcc, fps, size)
                    out2 = cv2.VideoWriter(self.output_path2, fourcc, fps, size)

                # split & write each half
                half_w = combined_frame.shape[1] // 2
                out1.write(combined_frame[:, :half_w])
                out2.write(combined_frame[:, half_w:])

                frame_idx += 1
                total_lines_drawn += lines_drawn

            print(f"Combined output saved to {self.output_path}")
            print(f"Single outputs saved to {self.output_path1} and {self.output_path2}")
            print(f"Frames processed: {frame_idx}, Lines drawn: {total_lines_drawn}")

        finally:
            if self.cap1:  self.cap1.release()
            if self.cap2:  self.cap2.release()
            if self.out:   self.out.release()
            if out1:       out1.release()
            if out2:       out2.release()
        def render_from_json(video_path, json_path, output_path):
            cap = cv2.VideoCapture(video_path)
            
            with open(json_path) as f:
                tracking_data = json.load(f)

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                cap.get(cv2.CAP_PROP_FPS),
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Get tracking data for current frame
                tracks = tracking_data.get(str(frame_idx), [])
                color_map =  {}
                for track in tracks:
                    x1, y1, x2, y2 = track["bbox"]
                    track_id = track['track_id']
                    color = self.get_color(track_id)
                    color_map[track_id] = color

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    display_id = f"{track_id:02d}" if track_id == 0 else str(track_id)
                    cv2.putText(frame, f"ID:{display_id}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    

                writer.write(frame)
                frame_idx += 1

            cap.release()
            writer.release()
            print(f"Video saved to: {output_path}")
        render_from_json(self.video1_path, self.json1_path.replace('.json', '_updated.json'),  self.output_path1 )
        render_from_json(self.video2_path, self.json2_path.replace('.json', '_updated.json'),  self.output_path2)     




