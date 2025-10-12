import requests

# Perplexity API endpoint (example, replace with actual)
API_URL = "https://api.perplexity.ai/v1/image/geolocation"

# Your API key
API_KEY = "YOUR_API_KEY_HERE"

# SAS URL of the input image
image_url = "https://sadronevideo.blob.core.windows.net/input/2/images/frame0.jpg?sp=racwdli&st=2025-09-08T00:09:00Z&se=2025-09-25T08:24:00Z&spr=https&sv=2024-11-04&sr=c&sig=8deCTxBW2FWK8WsjEVsCdsGDjIehlW%2FphKTehIyzDbY%3D"


def geolocation(image_url):
    # Prepare headers and payload
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "image_url": image_url
    }

    # Send request to Perplexity API
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        latitude = result.get("latitude", "")
        longitude = result.get("longitude", "")
        print(f"Estimated GPS coordinates: Latitude={latitude}, Longitude={longitude}")
        return latitude, longitude
    else:
        print("Error:", response.status_code, response.text)
        return "", ""

res = str(geolocation(image_url))
print(res)
   