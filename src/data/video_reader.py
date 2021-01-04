import numpy as np
import cv2

class VideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.t = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.x = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.y = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.codec = self.cap.get(cv2.CAP_PROP_FOURCC)
        
    def row_idx(self, edge):
        return int(self.x*edge)
    
    def col_idx(self, edge):
        return int(self.y*edge)
        
    def get_frames(self, frame_range=None, crop_box=None):
        if crop_box is not None:
            x_range = tuple(map(self.row_idx, (crop_box[0], crop_box[2])))
            y_range = tuple(map(self.col_idx, (crop_box[1], crop_box[3])))
        else:
            x_range = (0, self.x)
            y_range = (0, self.y)
            
        width = x_range[1]-x_range[0]
        height = y_range[1]-y_range[0]
            
        f_range = (-1, self.t) if frame_range is None else frame_range
        duration = self.t if frame_range is None else f_range[-1]-f_range[0]
        frames = np.empty((duration, height, width, 3), np.dtype('uint8'))
        
        fc = 0
        ret = True

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f_range[0])
        while (fc < duration and ret):
            ret, frame = self.cap.read()
            frames[fc] = frame[slice(*x_range)][slice(*y_range)]
            fc += 1

        return frames
    
    def get_frame(self, n):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, n-1)
        res, frame = self.cap.read()
        return frame
    
    def close(self):
        self.cap.release()
        cv2.waitKey(0)
        
    def save_frames(self, frames, path):
        writer = cv2.VideoWriter(
            path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            int(self.fps), 
            (frames.shape[2], frames.shape[1]), 
            True
        )
        
        for f in frames:
            writer.write(f)