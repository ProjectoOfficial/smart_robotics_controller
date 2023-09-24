import numpy as np
import cv2
import matplotlib.pyplot as plt


def feature_extractor(image, detector, mask=None):
    """
    provide keypoints and descriptors

    :params image: real time image

    """

    keypoints, descriptors = detector.detectAndCompute(image, mask)

    return keypoints, descriptors


def get_matcher(name:str):
    if name.lower() == "bf":
        return cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)
    elif name.lower() == "flann":
        return cv2.FlannBasedMatcher()
    else: 
        raise NotImplementedError
        

def get_detector(name:str):
    if name.lower() == "orb":
        return cv2.ORB.create() 
    elif name.lower() == "sift":
        return cv2.SIFT.create()
    elif name.lower() == "surf":
        return cv2.xfeatures2d.SURF_create(400)
    else:
        raise NotImplementedError


def feature_matching(feature_matcher:object, first_descriptor:np.ndarray, second_descriptor:np.ndarray, k:int=2, distance_threshold:float=1.0):
    """
    Match features between two images

    """
    filtered_matches = []
    
    if isinstance(feature_matcher, cv2.BFMatcher):
        matches = feature_matcher.knnMatch(first_descriptor, second_descriptor, k=k)
        
        for match1, match2 in matches:
            if match1.distance <= distance_threshold * match2.distance:
                filtered_matches.append(match1)
        
    elif isinstance(feature_matcher, cv2.FlannBasedMatcher):
        matches = feature_matcher.knnMatch(first_descriptor, second_descriptor, k=k)
        
        for match_pair in matches:
            if len(match_pair) == 2 and match_pair[0].distance < distance_threshold * match_pair[1].distance:
                filtered_matches.append(match_pair[0])
    else: 
        raise NotImplementedError
        
    
    return filtered_matches
        

def visualize_matches(first_image, second_image, keypoint_one, keypoint_two, matches):
    """
    Visualize corresponding matches in two images

    """
    show_matches = cv2.drawMatches(first_image, keypoint_one, second_image, keypoint_two, matches, None, flags=2)
    plt.figure(figsize=(15, 5), dpi=100)
    plt.imshow(show_matches)
    plt.show()