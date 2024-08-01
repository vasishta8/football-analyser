from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import os
import pickle
import sys
sys.path.append('../')
from utilities import get_center, get_width


class Tracker():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        detections = []
        batch_size = 16
        for i in range(0, len(frames), batch_size):
            batch_detection = self.model.predict(frames[i:i+batch_size], conf=0.1, verbose=False)
            detections += batch_detection
        return detections

    def get_object_track(self, frames, cached=False, cache_path = None):
        if cached and cache_path is not None and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks
        tracks = {
            "player": [],
            "ball": [],
            "referee": []
        }
        detections = self.detect_frames(frames)
        for frame_number, detection in enumerate(detections):
            tracks["player"].append({})
            tracks["ball"].append({})
            tracks["referee"].append({})
            cls_names = detection.names
            cls_names_map = {v:k for k,v in cls_names.items()}
            supervision_detection = sv.Detections.from_ultralytics(detection)
            for idx, class_id in enumerate(supervision_detection.class_id):
                if cls_names[class_id] == "goalkeeper":
                    supervision_detection.class_id[idx] = cls_names_map["player"]
            tracking_detection = self.tracker.update_with_detections(supervision_detection)
            for frame in tracking_detection:
                bound = frame[0].tolist()
                cls_id = frame[3]
                track_id = frame[4]
                if cls_id == cls_names_map["player"]:
                    tracks["player"][frame_number][track_id] = {"bound": bound}
                elif cls_id == cls_names_map["referee"]:
                    tracks["referee"][frame_number][track_id] = {"bound": bound}
            for frame in supervision_detection:
                bound = frame[0].tolist()
                cls_id = frame[3]
                if cls_id == cls_names_map["ball"]:
                    tracks["ball"][frame_number][1] = {"bound": bound}
        if cache_path is not None:
            with open(cache_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks
    
    def draw_ellipse(self, frame, bound, clr):
        cy = int(bound[3])
        cx, _ = get_center(bound)
        wd = get_width(bound)
        cv2.ellipse(
            frame,
            center = (cx, cy),
            axes = (int(wd), int(wd*0.4)),
            angle=0.0,
            startAngle=-45,
            endAngle=225,
            color=clr,
            thickness=2,
            lineType=cv2.LINE_4
        )
        return frame

    def draw_triangle(self, frame, bound, clr):
        cx, _ = get_center(bound)
        cy = int(bound[1])
        v1 = [cx-5, cy-15]
        v2 = [cx+5, cy-15]
        v3 = [cx, cy]
        points = np.array([v1,v2,v3])
        cv2.fillPoly(frame, [points], clr)
        return frame

    def draw_rectangle(self, frame, bound, clr, player_id):
        cx, cy = get_center(bound)
        wd = get_width(bound)
        tl = (int(bound[0]), int((bound[3])+wd*0.2))
        br = (int(bound[2]), int((bound[3])+15+wd*0.2))
        cv2.rectangle(frame, tl, br, clr, -1)
        cv2.putText(frame, str(player_id), (int(cx), int(bound[3])+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        return frame

    def draw_annotations(self, tracks, frames):
        output_frames = []
        for frame_number, frame in enumerate(frames):
            frame = frame.copy()
            player_dict = tracks["player"][frame_number]
            ball_dict = tracks["ball"][frame_number]
            referee_dict = tracks["referee"][frame_number]
            for track_id, box in player_dict.items():
                color = box.get("team_color", (0,0,255))
                frame = self.draw_ellipse(frame, box["bound"], color)
                frame = self.draw_rectangle(frame, box["bound"], color, track_id)
            for track_id, box in referee_dict.items():
                frame = self.draw_ellipse(frame, box["bound"], (0,255,255))
            for track_id, box in ball_dict.items():
                frame = self.draw_triangle(frame, box["bound"], (255,0,0))
            output_frames.append(frame)
        return output_frames
