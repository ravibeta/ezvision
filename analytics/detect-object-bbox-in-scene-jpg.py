import cv2
import numpy as np
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# ---------- Azure AI Search: retrieve vectors & basic similarity ----------

SEARCH_ENDPOINT = "https://<your-search-service>.search.windows.net"
SEARCH_INDEX = "<your-index-name>"
SEARCH_KEY = "<your-admin-or-query-key>"

# Assume index schema roughly:
# {
#   "name": "images",
#   "fields": [
#       { "name": "id", "type": "Edm.String", "key": true },
#       { "name": "description", "type": "Edm.String" },
#       {
#         "name": "contentVector",
#         "type": "Collection(Edm.Single)",
#         "searchable": True,
#         "dimensions": 1536,
#         "vectorSearchProfile": "myVectorProfile"
#       }
#   ]
# }
# Dimensions 1536 match common embedding models like text-embedding-ada-002 or text-embedding-3-small.[web:17][web:23]

def get_search_client():
    return SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX,
        credential=AzureKeyCredential(SEARCH_KEY),
    )

def get_image_docs():
    """
    Return the two documents from the index:
    one for the object image, one for the scene image.
    Assumes only two docs exist in the index.
    """
    client = get_search_client()
    results = client.search(
        search_text="*",  # match all
        top=2,
        vector=None  # not needed since there are only 2 docs
    )
    docs = list(results)
    if len(docs) != 2:
        raise RuntimeError("Expected exactly 2 documents (object + scene).")
    return docs

def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

# Example: validate that object/scene are semantically related via vectors
def validate_vectors():
    docs = get_image_docs()
    doc1, doc2 = docs[0], docs[1]
    sim = cosine_similarity(doc1["contentVector"], doc2["contentVector"])
    print(f"Vector cosine similarity between images: {sim:.4f}")
    # Use this as a sanity check; high similarity suggests the scene contains the object at semantic level.

# ---------- Local feature matching + homography for bounding box ----------

def detect_object_bounding_box(object_img_path, scene_img_path, min_match_count=10):
    """
    Detect the object in the scene regardless of position, orientation, and scale,
    returning the bounding box as (x_min, y_min, x_max, y_max) in scene coordinates.

    Uses SIFT keypoints + FLANN matcher + RANSAC homography as in standard OpenCV tutorials.[web:15][web:21][web:24]
    """
    # Read images
    obj_img = cv2.imread(object_img_path, cv2.IMREAD_GRAYSCALE)
    scene_img = cv2.imread(scene_img_path, cv2.IMREAD_GRAYSCALE)

    if obj_img is None or scene_img is None:
        raise ValueError("Could not load object or scene image.")

    # Create SIFT detector (OpenCV-contrib must be installed)
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp_obj, des_obj = sift.detectAndCompute(obj_img, None)
    kp_scene, des_scene = sift.detectAndCompute(scene_img, None)

    if des_obj is None or des_scene is None:
        raise RuntimeError("Failed to compute SIFT descriptors for one of the images.")

    # FLANN-based matcher for SIFT descriptors
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_obj, des_scene, k=2)

    # Lowe's ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Good matches: {len(good_matches)}")

    if len(good_matches) < min_match_count:
        print("Not enough matches for reliable homography.")
        return None

    # Extract matched keypoints
    src_pts = np.float32([kp_obj[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("Homography could not be computed.")
        return None

    # Get object image corners and transform them into scene coordinates
    h_obj, w_obj = obj_img.shape
    obj_corners = np.float32([
        [0, 0],
        [w_obj, 0],
        [w_obj, h_obj],
        [0, h_obj]
    ]).reshape(-1, 1, 2)

    scene_corners = cv2.perspectiveTransform(obj_corners, H)

    # Compute axis-aligned bounding box around the transformed corners
    xs = scene_corners[:, 0, 0]
    ys = scene_corners[:, 0, 1]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    bbox = (x_min, y_min, x_max, y_max)
    print(f"Detected bounding box in scene: {bbox}")
    return bbox, scene_corners, H

def draw_bounding_box(scene_img_path, bbox, output_path="scene_with_box.jpg"):
    """
    Draw axis-aligned bounding box on the scene image and save result.
    """
    img = cv2.imread(scene_img_path)
    if img is None:
        raise ValueError("Could not load scene image.")

    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    cv2.imwrite(output_path, img)
    print(f"Saved scene with bounding box to {output_path}")

# ---------- Optional: HDBSCAN step on matched points (experimental) ----------

def cluster_matched_points_with_hdbscan(src_pts, dst_pts):
    """
    Optional advanced idea: cluster destination matched points using HDBSCAN to
    robustly localize the dominant cluster of matches corresponding to the object.

    Requires: pip install hdbscan
    This is more useful when there are many objects or clutter; with a single object,
    SIFT+homography is usually enough.[web:22]
    """
    import hdbscan

    # Use destination points as clustering inputs
    pts = dst_pts.reshape(-1, 2)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
    labels = clusterer.fit_predict(pts)

    # Filter to the largest non-noise cluster
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        return None

    # Find label with maximum count
    counts = np.bincount(valid_labels)
    best_label = np.argmax(counts)

    cluster_pts = pts[labels == best_label]
    x_min, y_min = cluster_pts.min(axis=0)
    x_max, y_max = cluster_pts.max(axis=0)
    bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
    return bbox

# ---------- Example usage ----------

if __name__ == "__main__":
    # 1. Optionally validate/inspect the vectors from Azure AI Search
    validate_vectors()  # sanity check using the two 1536-dim vectors

    # 2. Localize object in scene at pixel level
    object_img_path = "object.jpg"
    scene_img_path = "scene.jpg"
    result = detect_object_bounding_box(object_img_path, scene_img_path)

    if result is not None:
        bbox, scene_corners, H = result
        draw_bounding_box(scene_img_path, bbox, output_path="scene_with_object_box.jpg")
    else:
        print("Object could not be reliably located in the scene.")
