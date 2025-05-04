import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.ai.openai import OpenAIClient

# Azure configuration
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT") or "https://<your-search-resource>.search.windows.net"
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY") or "<your-search-api-key>"
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
OPENAI_ENDPOINT = os.getenv("AZURE_OPEN_AI_ENDPOINT") or "https://<your-openai-resource>.openai.azure.com"
OPENAI_API_KEY = os.getenv("AZURE_OPEN_AI_API_KEY") or "<your-openai-api-key>"
OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPEN_AI_DEPLOYMENT_NAME") or "<your-deployment-name>"

# Initialize Azure Search client
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

# Initialize Azure OpenAI client
openai_client = OpenAIClient(
    endpoint=OPENAI_ENDPOINT,
    credential=AzureKeyCredential(OPENAI_API_KEY)
)

def search_images(query_text):
    """Search for images matching the input text in Azure AI Search."""
    results = search_client.search(
        search_text=query_text,
        query_type=QueryType.SEMANTIC,
        select=["image_url", "description"]
    )
    return [{"image_url": result["image_url"], "description": result["description"]} for result in results]

def ask_openai(images, query_text):
    """Ask Azure OpenAI model about objects detected in the images."""
    prompt = f"The following images were retrieved based on the text '{query_text}':\n"
    for image in images:
        prompt += f"- Image URL: {image['image_url']}, Description: {image['description']}\n"
    prompt += "How many objects were detected in these images that correspond to the given text?"

    response = openai_client.completions.create(
        deployment_id=OPENAI_DEPLOYMENT_NAME,
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].text.strip()

def main():
    query_text = input("Enter the text to search for matching images: ")
    images = search_images(query_text)
    if not images:
        print("No matching images found.")
        return

    response = ask_openai(images, query_text)
    print("OpenAI Response:", response)

if __name__ == "__main__":
    main()