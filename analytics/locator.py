from geospyer import GeoSpy
import os
gemini_api_key = os.getenv("GEMINI_API_KEY").strip('"')
if gemini_api_key != "AIzaSyC72ZYI-VzkyDnKi5oL4T77ArD1l1MZIpA":
   print(f"not a match for {gemini_api_key}")
else:
   print(f"is a match for {gemini_api_key}")
def get_nearest_latitude_longitude(image_path="frame5.jpg"):
    # Initialize GeoSpy with your Gemini API key
    geospy = GeoSpy(api_key=gemini_api_key)

    # Analyze the image
    result = geospy.locate(image_path=image_path)

    # Check for errors
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        # Extract location info
        if "locations" in result and result["locations"]:
            location = result["locations"][0]
            lat = location["coordinates"]["latitude"]
            lon = location["coordinates"]["longitude"]
            print(f"Estimated Coordinates: Latitude = {lat}, Longitude = {lon}")
            # Optional: Open in Google Maps
            # import webbrowser
            maps_url = f"https://www.google.com/maps?q={lat},{lon}"
            print(maps_url)
            #webbrowser.open(maps_url)
            return lat, lon
        else:
            print("No location data found.")
            return None, None

    # output:
    # Estimated Coordinates: Latitude = 42.3736, Longitude = -71.1097 or paste 42.37131916058968, -71.11736575063918 in Google Maps
    # https://www.google.com/maps?q=42.37131916058968, -71.11736575063918
    # (42.3736, -71.1097)
print(get_nearest_latitude_longitude())
# https://www.google.com/maps?q=42.3705793,-71.1202623
# https://maps.app.goo.gl/cFBvck4QGBrpHCiA6
