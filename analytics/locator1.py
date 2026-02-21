import nyckel
import os
nyckel_client_id = os.getenv("NYCKEL_CLIENT_ID")
nyckel_client_secret = os.getenv("NYCKEL_CLIENT_SECRET")
credentials = nyckel.Credentials(nyckel_client_id, nyckel_client_secret)
image_url = os.getenv("CIRCULAR_BUILDING_SAS_URL").strip('"')
response = nyckel.invoke("landmark-identifier", image_url, credentials)
print(response)
# Output:
# {'labelName': 'Yellowstone National Park', 'labelId': 'label_wottnvl9ole6ch4o', 'confidence': 0.02}
