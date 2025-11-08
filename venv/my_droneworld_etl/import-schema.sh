# Variables
search_service="srch-vision-01"
index_name="index007"
dest_index_name="$index_name"copy
resource_group="rg-ctl-2"
storage_account="sadronevideo"
container_name="metadata"
total_docs=2
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

exists=$(az storage blob exists \
  --account-name $storage_account \
  --account-key $storage_key \
  --container-name $container_name \
  --name $blob_name \
  --query exists --output tsv --only-show-errors)

if [ "$exists" != "true" ]; then
  echo "Skipping import for schema $blob_name (blob missing)"
  exit
fi

file_name="$index_name"-schema.json
echo $file_name
# Download blob
az storage blob download \
  --account-name $storage_account \
  --account-key $storage_key \
  --container-name $container_name \
  --name $blob_name \
  --file $file_name  \
  -o none

schema_exists=$(curl -X GET "https://$search_service.search.windows.net/indexes/$dest_index_name?api-version=2023-10-01-Preview" \
  -H "api-key: $admin_key" \
  -H "Content-Type: application/json" \
    | jq -r 'if .error then "false" else "true" end')

if [ "$exists_in_index" == "true" ]; then
  echo "Skipping import for schema (already exists in index)"
  rm "$file_name"
  continue
fi
sed -i "s/$index_name/$dest_index_name/g" "$file_name"
curl -X PUT "https://$search_service.search.windows.net/indexes/$dest_index_name?api-version=2023-10-01-Preview" \
  -H "api-key: $admin_key" \
  -H "Content-Type: application/json" \
  --data-binary "@$file_name"

echo "schema imported"
