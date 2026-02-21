import os
from urllib.parse import urlparse
scene_uri = os.getenv("AZURE_QUERY_SAS_URI").strip('"') 
def get_image_output_url(scene_uri):
    # Parse the original video URL to get account, container, and path
    parsed = urlparse(scene_uri)
    path_parts = parsed.path.split('/')
    blob_name = path_parts[-1].split('.')[0]
    print(f"blob_name={blob_name}")
    container = path_parts[1]
    blob_path = '/'.join(path_parts[2:])
    # Remove the file name from the blob path
    blob_dir = '/'.join(blob_path.split('/')[:-1])
    print(f"blob_dir={blob_dir}")
    if blob_dir == "" or blob_dir == None:
        blob_dir = "output"
    # Create image path
    image_path = f"{blob_dir}/analyzed/vehiclesframe.jpg"
    # Rebuild the base URL (without SAS token)
    base_url = f"{parsed.scheme}://{parsed.netloc}/{container}/{image_path}"
    # Add the SAS token if present
    sas_token = parsed.query
    if sas_token:
        image_url = f"{base_url}?{sas_token}"
    else:
        image_url = base_url
    return image_url

print(scene_uri)
print(get_image_output_url(scene_uri))