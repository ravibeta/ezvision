import os
from openai import AzureOpenAI

endpoint = "https://openvision.openai.azure.com/"
model_name = "gpt-4o-mini"
deployment = "gpt-4o-mini"

subscription_key = "<your-api-key>"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": """
You are an AI assistant that answers questions about the stored and indexed drone images and objects in search index index02.
The data source is an Azure AI Search resource where the schema has JSON description field, a vector field and an id field and this id field must be cited in your answer.
If you do not find a match for the query, respond with "I don't know", otherwise cite references with the value of the id field.
""",
        },
        {
            "role": "user",
            "content": "How many red cars can be found near the building with a roof that has a circular structure?",
        }
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=deployment
)

print(response.choices[0].message.content)