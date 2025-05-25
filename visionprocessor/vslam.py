import cv2
import numpy as np


# Function to detect and extract features from the aerial images

def extract_features(image_path): 
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    orb = cv2.ORB_create() 
    keypoints, descriptors = orb.detectAndCompute(image, None) 
    return keypoints, descriptors, image 

def match_features(descriptors1, descriptors2): 
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
    matches = matcher.match(descriptors1, descriptors2) 
    matches = sorted(matches, key=lambda x: x.distance)  # Sort by match quality 
    return matches 


def get_matches(image1_file, image2_file):
    # Extract features from both images 
    keypoints1, descriptors1, image1 = extract_features(image1_file) 
    keypoints2, descriptors2, image2 = extract_features(image2_file) 
    matches = match_features(descriptors1, descriptors2) 
    return matches


def get_matches_image(image1_file, image2_file):
    # Extract features from both images 
    keypoints1, descriptors1, image1 = extract_features(image1_file) 
    keypoints2, descriptors2, image2 = extract_features(image2_file) 
    matches = match_features(descriptors1, descriptors2)
    output_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
    return output_image
    
# sample invocation
#output_image = get_matches_image("000001.jpg", "000002.jpg")
#cv2.imwrite("output_image_1.jpg", output_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#sample output
# https://tinyurl.com/orbimage