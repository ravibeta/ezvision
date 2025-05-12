import sift
import cv2
import json
import math
import multiprocessing

def stable_groups(keypoints, groups, threshold):
    for kp in keypoints:
        matched = false
        for group in groups:
            mean_feature = get_mean_feature(group)
            recent_pixel = get_recent_pixel(group)
            if kd.feature - mean_feature < threshold and abs(lucas_kanade_optical_flow(recent_pixel)-kp.pixel) < threshold:
               group.add(kp)
               matched = true
               break
        if matched == false:
            groups.add(create_group(kp))
def global_groups(stable_groups,global_groups, threshold):
    for stable_group in stable_groups:
        matched = false
        for global_group in global_groups:
            mean_feature = get_mean_feature(global_group)
            recent_pixel = get_recent_pixel(global_group)
            if get_mean_feature(stable_group) - mean_feature < threshold and delta_least_squares(stable_group,global_group):
               global_group.add(stable_group)
               matched = true
               break
        if matched == false:
            global_groups.add(create_global_group(stable_group))

def get_recent_pixel(bboxes: list):
    """
    Finds the bounding box closest to the center of all given bounding boxes.

    :param bboxes: List of tuples containing bounding box coordinates and a float value.
                   Each tuple is ((x_min, y_min, x_max, y_max), value).
    :return: Tuple representing the bounding box closest to the center of all bounding boxes.
    """
    if not bboxes:
        return None

    # Compute the overall bounding box center
    all_x_mins = [bbox[0][0] for bbox in bboxes]
    all_y_mins = [bbox[0][1] for bbox in bboxes]
    all_x_maxs = [bbox[0][2] for bbox in bboxes]
    all_y_maxs = [bbox[0][3] for bbox in bboxes]

    center_x = (min(all_x_mins) + max(all_x_maxs)) / 2
    center_y = (min(all_y_mins) + max(all_y_maxs)) / 2

    def bbox_distance(bbox):
        x_min, y_min, x_max, y_max = bbox[0]
        bbox_center_x = (x_min + x_max) / 2
        bbox_center_y = (y_min + y_max) / 2
        return ((bbox_center_x - center_x) ** 2 + (bbox_center_y - center_y) ** 2) ** 0.5

    return min(bboxes, key=bbox_distance)

def get_mean_feature(bboxes: list):
    """
    Computes the mean of all values associated with bounding boxes.

    :param bboxes: List of tuples containing bounding box coordinates and a float value.
                   Each tuple is ((x_min, y_min, x_max, y_max), value).
    :return: Mean of the float values.
    """
    if not bboxes:
        return 0.0

    return sum(value for _, value in bboxes) / len(bboxes)



LK_PARAMETERS = dict(winSize=(21, 21), maxLevel=2, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01))

def homography_from_flow(prev_homography, prev_gray, cur_gray):
	positions = []
	for i in range(0, prev_gray.shape[0]-50, 50):
		for j in range(0, prev_gray.shape[1]-50, 50):
			positions.append((i, j))
	positions_np = numpy.array(positions, dtype='float32').reshape(-1, 1, 2)

	def flip_pos(positions):
		return numpy.stack([positions[:, :, 1], positions[:, :, 0]], axis=2)

	next_positions, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, flip_pos(positions_np), None, **LK_PARAMETERS)
	if next_positions is None:
		return None

	next_positions = flip_pos(next_positions)
	differences = next_positions[:, 0, :] - positions_np[:, 0, :]
	differences_okay = differences[numpy.where(st[:, 0] == 1)]
	median = [numpy.median(differences_okay[:, 0]), numpy.median(differences_okay[:, 1])]
	good = (numpy.square(differences[:, 0] - median[0]) + numpy.square(differences[:, 1] - median[1])) < 16

	if float(numpy.count_nonzero(good)) / differences.shape[0] < 0.7:
		return None

	# translate previous homography based on the flow result
	translation = [numpy.median(differences[:, 0]), numpy.median(differences[:, 1])]
	H_translation = numpy.array([[1, 0, -translation[1]], [0, 1, -translation[0]], [0,0,1]], dtype='float32')
	return prev_homography.dot(H_translation)

