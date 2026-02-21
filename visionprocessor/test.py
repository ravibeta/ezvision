import os
from PIL import Image
from landingai.predict import Predictor

# Enter your API Key
endpoint_id = "11cb6c44-3b6a-4b47-bac9-031826bc80ea"
api_key = os.getenv("LANDING_AI_API_KEY") or "YOUR_API_KEY"

# Load your image
image = Image.open("image.jpg")

# Run inference
predictor = Predictor(endpoint_id, api_key=api_key)
predictions = predictor.predict(image)