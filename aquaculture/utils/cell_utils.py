import numpy as np
import cv2

def init_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = 1
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    return detector
    pass  # init_blob_detector


def detect_cells(image_path, verbose=1):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    detector = init_blob_detector()
    blobs = detector.detect(gray)

    if verbose == 1:
        image_blob = cv2.drawKeypoints(image, blobs,
                                       np.array([]), (255, 0, 0),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return blobs, image_blob

    return blobs, None
    pass # detect_cells
