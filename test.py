import cv2
import numpy as np
from utils import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="test",
        description="Play a video from a dicom file."
    )
    parser.add_argument(
        'path',
        type=str,
        help="Path to dicom file"
    )
    args = parser.parse_args()

    # Process data
    path = args.path
    ds = read_dicom(path)
    arr = ds.pixel_array
    frame_time = ds.FrameTime

    i = 0
    while True:
        frame = arr[i%arr.shape[0]]
        frame = cv2.normalize(frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow('Video', frame)
        i+=1
        # esc key
        if cv2.waitKey(int(frame_time)) & 0xFF == 27: 
            break
    cv2.destroyAllWindows()