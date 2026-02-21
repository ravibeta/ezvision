import json
import numpy as np

# Replace this with actual GPS bounds for transformation
# Example: top-left, top-right, bottom-right, bottom-left in pixel & GPS
pixel_bounds = np.array([[0, 0], [4096, 0], [4096, 4096], [0, 4096]])
gps_bounds = np.array([[39.735, -104.997], [39.735, -104.989],
                       [39.729, -104.989], [39.729, -104.997]])

# Compute affine transform matrix from pixel to GPS
A = np.linalg.lstsq(pixel_bounds, gps_bounds, rcond=None)[0]

def pixel_to_gps(coord):
    """Map pixel coordinate to GPS using affine approximation"""
    return tuple(np.dot(coord, A))

def parse_json_gps(json_data):
    gps_coords = []
    for frame in json_data:
        if frame is None:
            continue
        frame_coords = [pixel_to_gps(coord) for coord in frame]
        gps_coords.append(frame_coords)
    return gps_coords

# Example JSON input
data = [None, [[3132, 4151], [3354, 2924], [4044, 3056], [3824, 4275]],
              [[3095, 4164], [3318, 2939], [4006, 3073], [3787, 4289]]]

gps_output = parse_json_gps(data)
for i, frame in enumerate(gps_output):
    print(f"Frame {i+1}:")
    for lat, lon in frame:
        print(f"Latitude: {lat:.6f}, Longitude: {lon:.6f}")
