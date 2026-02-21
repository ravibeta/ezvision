import json
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import os
import re
search_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
api_version = os.getenv("AZURE_SEARCH_API_VERSION")
search_api_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "index00")
credential = AzureKeyCredential(search_api_key)
target_id = "003401" 

# Initialize SearchClient
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_api_key)
)

def prepare_json_string_for_load(text):
  text = text.replace("\"", "'")
  text = text.replace("{'", "{\"")
  text = text.replace("'}", "\"}")
  text = text.replace(" '", " \"")
  text = text.replace("' ", "\" ")
  text = text.replace(":'", ":\"")
  text = text.replace("':", "\":")
  text = text.replace(",'", ",\"")
  text = text.replace("',", "\",")
  return re.sub(r'\n\s*', '', text)
  
def to_string(bounding_box):
    return f"{bounding_box['x']},{bounding_box['y']},{bounding_box['w']},{bounding_box['h']}"
    
# Retrieve the first 10 entries from the index
entry = search_client.get_document(key=target_id) # , select=["id", "description"])
print(entry["id"])
entry["description"] = '{"_data":{"captionResult":{"text":"aerial view of a street and buildings","confidence":0.6662805080413818},"denseCaptionsResult":{"values":[{"text":"aerial view of a street and buildings","confidence":0.6662805080413818,"boundingBox":{"x":0,"y":0,"w":1920,"h":1080}},{"text":"a top view of a building","confidence":0.6719546914100647,"boundingBox":{"x":871,"y":535,"w":295,"h":527}},{"text":"a green curtain with white stripes","confidence":0.5812939405441284,"boundingBox":{"x":1289,"y":919,"w":180,"h":156}},{"text":"a blurry image of a white ball","confidence":0.7682946920394897,"boundingBox":{"x":1785,"y":402,"w":55,"h":57}},{"text":"a white square on a gray surface","confidence":0.7140303254127502,"boundingBox":{"x":1531,"y":356,"w":58,"h":58}},{"text":"a close-up of a concrete wall","confidence":0.7894774079322815,"boundingBox":{"x":886,"y":300,"w":1015,"h":238}},{"text":"a black square in a black box","confidence":0.6790308952331543,"boundingBox":{"x":1576,"y":626,"w":80,"h":79}},{"text":"aerial view of a street with cars and buildings","confidence":0.7034556269645691,"boundingBox":{"x":0,"y":0,"w":1891,"h":1047}},{"text":"a close-up of a machine","confidence":0.7010144591331482,"boundingBox":{"x":896,"y":542,"w":247,"h":197}},{"text":"a blurry image of people standing on ice","confidence":0.7135794162750244,"boundingBox":{"x":309,"y":1011,"w":229,"h":65}}]},"metadata":{"width":1920,"height":1080},"modelVersion":"2023-10-01","objectsResult":{"values":[]},"peopleResult":{"values":[{"boundingBox":{"x":1184,"y":1,"w":28,"h":56},"confidence":0.04930095747113228},{"boundingBox":{"x":257,"y":353,"w":24,"h":66},"confidence":0.010437356308102608},{"boundingBox":{"x":417,"y":1035,"w":29,"h":42},"confidence":0.009571903385221958},{"boundingBox":{"x":1787,"y":414,"w":46,"h":46},"confidence":0.006558298598974943},{"boundingBox":{"x":1565,"y":277,"w":37,"h":38},"confidence":0.004671717528253794},{"boundingBox":{"x":1515,"y":762,"w":44,"h":116},"confidence":0.0033057646360248327},{"boundingBox":{"x":279,"y":0,"w":171,"h":45},"confidence":0.0032794857397675514},{"boundingBox":{"x":1370,"y":332,"w":54,"h":123},"confidence":0.0030708452686667442},{"boundingBox":{"x":1210,"y":418,"w":36,"h":50},"confidence":0.0030129882507026196},{"boundingBox":{"x":1495,"y":54,"w":21,"h":55},"confidence":0.0030047486070543528},{"boundingBox":{"x":881,"y":116,"w":15,"h":24},"confidence":0.002573195146396756},{"boundingBox":{"x":1207,"y":0,"w":30,"h":46},"confidence":0.002285190625116229},{"boundingBox":{"x":954,"y":1,"w":23,"h":43},"confidence":0.0018366944277659059},{"boundingBox":{"x":941,"y":428,"w":35,"h":43},"confidence":0.0018296297639608383},{"boundingBox":{"x":1358,"y":395,"w":28,"h":97},"confidence":0.001794277923181653},{"boundingBox":{"x":290,"y":286,"w":37,"h":66},"confidence":0.001736101577989757},{"boundingBox":{"x":610,"y":466,"w":36,"h":86},"confidence":0.001492612762376666},{"boundingBox":{"x":449,"y":1043,"w":35,"h":34},"confidence":0.0013902209466323256},{"boundingBox":{"x":258,"y":348,"w":19,"h":30},"confidence":0.001335522043518722},{"boundingBox":{"x":843,"y":173,"w":19,"h":42},"confidence":0.0011864334810525179},{"boundingBox":{"x":1031,"y":142,"w":22,"h":35},"confidence":0.0010886145755648613},{"boundingBox":{"x":1402,"y":333,"w":22,"h":22},"confidence":0.0010595063213258982}]},"readResult":{"blocks":[{"lines":[{"text":"BUS","boundingPolygon":[{"x":751,"y":60},{"x":776,"y":61},{"x":775,"y":76},{"x":751,"y":76}],"words":[{"text":"BUS","boundingPolygon":[{"x":751,"y":60},{"x":776,"y":60},{"x":775,"y":76},{"x":751,"y":75}],"confidence":0.663}]},{"text":"ONLY CL_Y","boundingPolygon":[{"x":360,"y":150},{"x":358,"y":253},{"x":321,"y":253},{"x":318,"y":149}],"words":[{"text":"ONLY","boundingPolygon":[{"x":360,"y":149},{"x":360,"y":182},{"x":318,"y":183},{"x":318,"y":149}],"confidence":0.858},{"text":"CL_Y","boundingPolygon":[{"x":360,"y":191},{"x":360,"y":252},{"x":318,"y":253},{"x":318,"y":191}],"confidence":0.136}]}]}]},"smartCropsResult":{"values":[{"aspectRatio":1.78,"boundingBox":{"x":80,"y":45,"w":1760,"h":990}}]},"tagsResult":{"values":[{"name":"building","confidence":0.9955792427062988},{"name":"outdoor","confidence":0.9789592027664185},{"name":"house","confidence":0.9247981309890747},{"name":"window","confidence":0.9246839284896851},{"name":"neighbourhood","confidence":0.9180514216423035},{"name":"residential area","confidence":0.8790454864501953},{"name":"urban design","confidence":0.8641414046287537},{"name":"car","confidence":0.8590799570083618},{"name":"apartment","confidence":0.8530478477478027},{"name":"town","confidence":0.8485689163208008},{"name":"street","confidence":0.838438093662262},{"name":"city","confidence":0.7917639017105103},{"name":"apartment building","confidence":0.7477256059646606}]}},"description":"aerial view of a street and buildings,aerial view of a street and buildings,a top view of a building,a green curtain with white stripes,a blurry image of a white ball,a white square on a gray surface,a close-up of a concrete wall,a black square in a black box,aerial view of a street with cars and buildings,a close-up of a machine,a blurry image of people standing on ice"}'
# print(entry["description"])
merge_results = search_client.merge_documents([entry])
if merge_results:
	print(f"{merge_results[0].succeeded}")
	print(f"{merge_results[0].error_message}")