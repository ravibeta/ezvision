from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import ConnectionType
from dotenv import load_dotenv 
import os 
load_dotenv(override=True) 

project_endpoint = os.environ["AZURE_PROJECT_ENDPOINT"] 
project_client = AIProjectClient(endpoint=project_endpoint, credential=DefaultAzureCredential()) 
project_client.agents.delete_agent("asst_jHRIUH1c3MF4UyxBpRLXtDiu")
project_client.agents.delete_agent("asst_zRK6pXL4UGqJ3wC8cxLUUet6")
project_client.agents.delete_agent("asst_a9vWvxg2CpFpc8QRXSYrxk4S")
project_client.agents.delete_agent("asst_sENNxZSb8kA4C5k0bjs3N4W6")
project_client.agents.delete_agent("asst_ZmzLYYAVYljrXliIswjTPJnA")
# asst_f3IUTrON3hMpdyTJU51aCo7v
project_client.agents.delete_agent("asst_vGYWQVbDdQCnbgBjKGguJFbI")