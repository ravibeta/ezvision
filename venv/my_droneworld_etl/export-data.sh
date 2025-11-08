#!/bin/bash
# Export one document at a time using REST API and loop
# Variables
search_service="srch-vision-01"
index_name="index007"
resource_group="rg-ctl-2"
storage_account="sadronevideo"
container_name="metadata"
total_docs=27
api_version="2023-10-preview"

echo $search_service
echo $index_name
echo $resource_group
echo $storage_account
echo $container_name
echo $total_docs
# Get admin key
admin_key=$(az search admin-key show --service-name $search_service --resource-group $resource_group --query primaryKey --output tsv)
echo $admin_key
storage_key=$(az storage account keys list \
  --account-name $storage_account \
  --resource-group $resource_group \
  --query "[0].value" --output tsv)
echo $storage_key


for ((i=0; i<$total_docs; i++)); do
  file_name="doc_$i.json"
  blob_name="indexes/$index_name/data/$file_name"

  # Check if blob already exists
  exists=$(az storage blob exists \
    --account-name $storage_account \
    --account-key $storage_key \
    --container-name $container_name \
    --name $blob_name \
    --query exists --output tsv)

  if [ "$exists" == "true" ]; then
    echo "Skipping export for doc $i (already exists in blob)"
    continue
  fi

  # Export one document
  curl -s -X POST "https://$search_service.search.windows.net/indexes/$index_name/docs/search?api-version=2023-10-01-Preview" \
    -H "api-key: $admin_key" \
    -H "Content-Type: application/json" \
    -d "{\"search\":\"*\",\"top\":1,\"skip\":$i}" \
    | jq '.value[0]' > "$file_name"

  # Upload to blob
  az storage blob upload \
    --account-name $storage_account \
    --account-key $storage_key \
    --container-name $container_name \
    --name $blob_name \
    --file $file_name 

  # Clean up local file
  rm "$file_name"
done

