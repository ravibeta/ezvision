import requests
import os
from dotenv import load_dotenv
load_dotenv(override=True)
import time
from pprint import pprint
video_indexer_endpoint = os.getenv("AZURE_VIDEO_INDEXER_URL", "https://api.videoindexer.ai")
video_indexer_region = os.getenv("AZURE_VIDEO_INDEXER_REGION", "eastus")
video_indexer_account_id = os.getenv("AZURE_VIDEO_INDEXER_ACCOUNT")
video_Indexer_api_key = os.getenv("AZURE_VIDEO_INDEXER_API_KEY")
video_indexer_project = os.getenv("AZURE_VIDEO_INDEXER_PROJECT", "wfmat1ysct")
video_file_path = os.getenv("AZURE_VIDEO_INPUT", "mainindexedvideo.mp4")
access_token = os.getenv("AZURE_VIDEO_INDEXER_ACCESS_TOKEN")
video_id = os.getenv("AZURE_VIDEO_ID", "e474ubuwrr") # videosummary01
duration = os.getenv("AZURE_VIDEO_DURATION_IN_SECONDS", "307")

def get_access_token():
    """Retrieve an access token for the Video Indexer API."""
    url = f"{video_indexer_endpoint}/auth/{video_indexer_region}/Accounts/{video_indexer_account_id}/AccessToken"
    headers = {
        "Ocp-Apim-Subscription-Key": video_Indexer_api_key
    }
    response = requests.get(url, headers=headers)
    return response.text.strip('"')

def repeat_video_index(access_token, video_id):
    """Retrieve the index/insights for a video by its ID."""
    url = f"{video_indexer_endpoint}/{video_indexer_region}/Accounts/{video_indexer_account_id}/Videos/{video_id}/ReIndex?accessToken={access_token}"
    response = requests.put(url)
    if response.status_code == 200:
        return response
    return get_video_insights(access_token, video_id)
    
def get_video_insights(access_token, video_id):
    url = f"{video_indexer_endpoint}/{video_indexer_region}/Accounts/{video_indexer_account_id}/Videos/{video_id}/Index?accessToken={access_token}"
    count = 0
    while True:
        response = requests.get(url)
        data = response.json()
        if "state" in data and data['state'] == 'Processed':
            return data
        count+=1
        if count%10 == 0:
            print(data)
        print("Sleeping for ten seconds...")
        time.sleep(10)  # Wait 10 seconds before checking again

def get_selected_segments(insights, threshold):
        indexed_duration = insights["summarizedInsights"]["duration"]["seconds"]
        reduced_duration = (threshold * indexed_duration) / 100
        selected_segments = []
        # total_duration = 0
        for video in insights["videos"]:
            for shot in video["insights"]["shots"]:
                shot_id = shot["id"]
                for key_frame in shot["keyFrames"]:
                    key_frame_id = key_frame["id"]
                    start = key_frame["instances"][0]["start"]
                    end = key_frame["instances"][0]["end"]
                    # total_duration += float(end) - float(start)
                    print(f"Clipping shot: {shot_id}, key_frame: {key_frame_id}, start: {start}, end: {end}")
                    selected_segments +=[(start,end)]
        # print(f"Total duration: {total_duration}")
        return selected_segments

def create_project(access_token, video_id, selected_segments):
        import random
        import string
        video_ranges = []
        for start,end in selected_segments:
            intervals = {}
            intervals["videoId"] = video_id
            intervalRange = {}
            intervalRange["start"] = start
            intervalRange["end"] = end
            intervals["range"] = intervalRange
            video_ranges += [intervals]
        project_name = ''.join(random.choices(string.hexdigits, k=8))
        data = {
            "name": project_name,
            "videosRanges": video_ranges,
            "isSearchable": "false"
        }
        headers = {
            "Content-Type": "application/json"
        }
        url = f"{video_indexer_endpoint}/{video_indexer_region}/Accounts/{video_indexer_account_id}/Projects?accessToken={access_token}"
        response = requests.post(url, json=data, headers=headers)
        print(response.content)
        if response.status_code == 200:
            data = response.json()
            project_id = data["id"]
            return project_id
        else:
            return None
        
def render_video(access_token, project_id):
        url = f"{video_indexer_endpoint}/{video_indexer_region}/Accounts/{video_indexer_account_id}/Projects/{project_id}/render?sendCompletionEmail=false&accessToken={access_token}"
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers)
        print(response.content)
        if response.status_code == 202:
            return response
        else:
            return None
            
def get_render_operation(access_token, project_id):
    url = f"{video_indexer_endpoint}/{video_indexer_region}/Accounts/{video_indexer_account_id}/Projects/{project_id}/renderoperation?accessToken={access_token}"
    while True:
        response = requests.get(url)
        data = response.json()
        if "state" in data and data['state'] == 'Succeeded':
            return data
        print("Sleeping for ten seconds before checking on rendering...")
        time.sleep(10)  # Wait 10 seconds before checking again        

def download_rendered_file(access_token, project_id):
    url = f"{video_indexer_endpoint}/{video_indexer_region}/Accounts/{video_indexer_account_id}/Projects/{project_id}/renderedfile/downloadurl?accessToken={access_token}"
    response = requests.get(url)
    if response.status_code == 200:
        print(response.content)
        data = response.json()
        if "downloadUrl" in data:
            return data["downloadUrl"]
    return None
    
    
def get_timestamps(access_token, video_id):
    # Main workflow
    # access_token = get_access_token()
    insights = get_video_insights(access_token, video_id)
    #pprint(insights)
    timestamps=[]
    for keyframe in insights['videos'][0]['insights']['shots'][0]['keyFrames']:
        timestamps+=[(keyframe['instances'][0]['start'], keyframe['instances'][0]['end'])]
    print(timestamps)
    return timestamps
# [('0:00:00.2219828', '0:00:00.2570043'), ('0:00:00.5030307', '0:00:00.5379849'), ('0:00:00.8780307', '0:00:00.9249731')]
"""
selected_segments = get_selected_segments(insights, 10)
project_id = video_indexer_project
if not project_id:
    project_id = create_project(access_token, video_id, selected_segments)
print(project_id)
if project_id:
    render_response = render_video(access_token, project_id)
    print(render_response)
    if render_response:
        status = get_render_operation(access_token, project_id)
        print(status)
        download_url = download_rendered_file(access_token, project_id)
        print(download_url)
"""    
    

"""
{'state': 'Succeeded', 'result': {'videoRanges': [{'videoId': 'lwxjba8wy3', 'range': {'start': '0:00:02.7806641', 'end': '0:00:02.8139974'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:00:30.1820313', 'end': '0:00:30.2153646'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:00:48.25', 'end': '0:00:48.2840495'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:00:54.5133464', 'end': '0:00:54.5466797'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:00:56.5806641', 'end': '0:00:56.6139974'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:01:15.8330078', 'end': '0:01:15.8663411'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:01:24.333724', 'end': '0:01:24.3670573'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:01:35.2317057', 'end': '0:01:35.2650391'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:01:41.280013', 'end': '0:01:41.3270182'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:01:50.4290365', 'end': '0:01:50.4623698'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:01:51.5296875', 'end': '0:01:51.5630208'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:01:56.8983724', 'end': '0:01:56.9317057'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:02:07.4166667', 'end': '0:02:07.45'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:02:21.9847005', 'end': '0:02:22.0180339'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:02:39.0670573', 'end': '0:02:39.1139974'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:02:49.449349', 'end': '0:02:49.4826823'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:02:58.1507161', 'end': '0:02:58.1840495'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:03:08.3180339', 'end': '0:03:08.3513672'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:03:20.4186849', 'end': '0:03:20.4520182'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:03:24.8433594', 'end': '0:03:24.8766927'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:03:27.025', 'end': '0:03:27.0583333'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:03:58.1970052', 'end': '0:03:58.2303385'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:04:13.4656901', 'end': '0:04:13.4990234'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:04:26.5160156', 'end': '0:04:26.549349'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:04:40.05', 'end': '0:04:40.0833333'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:04:49.764388', 'end': '0:04:49.7977214'}}, {'videoId': 'lwxjba8wy3', 'range': {'start': '0:04:58.364388', 'end': '0:04:58.3977214'}}]}, 'error': None}
b'{"downloadUrl":"https://sadronevideo.blob.core.windows.net/vi-rendered-wfmat1ysct-0e4129/wfmat1ysct_rendered.mp4?skoid=c855c9a9-c3b8-449d-b876-75304f769177&sktid=1f4c33e1-e960-43bf-a135-6db8b82b6885&skt=2025-06-25T03%3A54%3A43Z&ske=2025-07-02T03%3A54%3A43Z&sks=b&skv=2021-10-04&sv=2021-10-04&st=2025-06-27T03%3A34%3A35Z&se=2025-06-27T04%3A39%3A35Z&sr=b&sp=r&scid=942959fd-c90a-4aab-8cd8-3ab7881559b8&sig=GkzcpywilXA74afVr%2BaUtcF08Tu%2Fz5X4cS0jpGeudIA%3D"}'
Result:
https://sadronevideo.blob.core.windows.net/vi-rendered-wfmat1ysct-0e4129/wfmat1ysct_rendered.mp4?skoid=c855c9a9-c3b8-449d-b876-75304f769177&sktid=1f4c33e1-e960-43bf-a135-6db8b82b6885&skt=2025-06-25T03%3A54%3A43Z&ske=2025-07-02T03%3A54%3A43Z&sks=b&skv=2021-10-04&sv=2021-10-04&st=2025-06-27T03%3A34%3A35Z&se=2025-06-27T04%3A39%3A35Z&sr=b&sp=r&scid=942959fd-c90a-4aab-8cd8-3ab7881559b8&sig=GkzcpywilXA74afVr%2BaUtcF08Tu%2Fz5X4cS0jpGeudIA%3D

and from the original for comparision
{
    "downloadUrl": "https://sadronevideo.blob.core.windows.net/vi-rendered-lmq98b93d3-322e35/lmq98b93d3_rendered.mp4?skoid=c855c9a9-c3b8-449d-b876-75304f769177&sktid=1f4c33e1-e960-43bf-a135-6db8b82b6885&skt=2025-06-27T01%3A01%3A58Z&ske=2025-07-04T01%3A01%3A58Z&sks=b&skv=2021-10-04&sv=2021-10-04&st=2025-06-27T02%3A18%3A16Z&se=2025-06-27T03%3A23%3A16Z&sr=b&sp=r&scid=1a69c336-6464-40ae-8691-7dacd43ab8bc&sig=MrWWeZpdktdNCykS8Ttg0tLO8Un4VhClqGhlLuPrjD0%3D"
}
and from the portal:
<source src="blob:https://www.videoindexer.ai/802c080c-9fae-41a2-9030-f464109a9d9e" type="">
"""

