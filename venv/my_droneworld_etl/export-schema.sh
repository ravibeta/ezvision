# Variables
search_service="srch-vision-01"
index_name="index02"
resource_group="rg-ctl-2"
schema_file=$(echo index-"$index_name"-schema.json)
echo $search_service
echo $index_name
echo $resource_group
echo $schema_file

# Get admin key
admin_key=$(az search admin-key show --service-name $search_service --resource-group $resource_group --query primaryKey --output tsv)
echo $admin_key
# Export schema using REST API
curl -X GET "https://$search_service.search.windows.net/indexes/$index_name?api-version=2023-10-01-Preview" \
  -H "api-key: $admin_key" \
  -H "Content-Type: application/json" \
  -o $schema_file
echo "schema exported"

