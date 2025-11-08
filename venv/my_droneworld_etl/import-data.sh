#!/bin/bash
# Export one document at a time using REST API and loop
# Variables
search_service="srch-vision-01"
index_name="index007"
dest_index_name="$index_name"copy
resource_group="rg-ctl-2"
storage_account="sadronevideo"
container_name="metadata"
total_docs=27
api_version="2023-10-preview"

echo $search_service
echo $index_name
echo $dest_index_name
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

  # Check if blob exists
  exists=$(az storage blob exists \
    --account-name $storage_account \
    --account-key $storage_key \
    --container-name $container_name \
    --name $blob_name \
    --query exists --output tsv)

  if [ "$exists" != "true" ]; then
    echo "Skipping import for doc $i (blob missing)"
    continue
  fi

  # Download blob
  az storage blob download \
    --account-name $storage_account \
    --account-key $storage_key \
    --container-name $container_name \
    --name $blob_name \
    --file $file_name  \
    -o none

  if [ ! -f "$file_name" ]; then
    echo "Skipping import for doc $i (download failed)"
    continue
  fi

  # Extract document ID
  doc_id=$(jq -r '.["@search.documentKey"] // .id // .Id // .ID' "$file_name")

  if [ -z "$doc_id" ]; then
    echo "Skipping import for doc $i (missing ID)"
    rm "$file_name"
    continue
  fi
  echo $doc_id

  # Check if document already exists in index
  exists_in_index=$(curl -s -X GET "https://$search_service.search.windows.net/indexes/$dest_index_name/docs/$doc_id?api-version=2023-10-01-Preview" \
    -H "api-key: $admin_key" \
    -H "Content-Type: application/json" \
    | jq -r 'if .error then "false" else "true" end')

  if [ "$exists_in_index" == "true" ]; then
    echo "Skipping import for doc $i (already exists in index)"
    rm "$file_name"
    continue
  fi

  # jq 'with_entries(select(.key != "id"))' "$file_name" > "filtered_$file_name"
  jq '{value: [.]}' "$file_name" > "filtered_$file_name"
  # Import to index
  curl -s -X POST "https://$search_service.search.windows.net/indexes/$dest_index_name/docs/index?api-version=2023-10-01-Preview" \
    -H "api-key: $admin_key" \
    -H "Content-Type: application/json" \
    --data-binary "@filtered_$file_name"

  # Clean up local file
  rm filtered_"$file_name"
  rm "$file_name"
done

