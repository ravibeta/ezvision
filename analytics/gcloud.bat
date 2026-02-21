rem gcloud services enable landmark.googleapis.com

rem gcloud iam service-accounts create landmark-client --description="Service account for Landmark API access"  --display-name="Landmark Client"

rem gcloud iam service-accounts keys create key.json  --iam-account=landmark-client@gcloudsdk01.iam.gserviceaccount.com

rem gcloud projects add-iam-policy-binding gcloudsdk01 --member="serviceAccount:landmark-client@gcloudsdk01.iam.gserviceaccount.com" --role="roles/cloudapis.serviceAgent"

rem gcloud services enable serviceusage.googleapis.com

gcloud services api-keys create --display-name="Landmark API Key"
