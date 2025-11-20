from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from .models import VideoEntity
from .serializers import VideoEntitySerializer
from .myvideoanalyzer import knowledge_base_search, run_function_tools, synthesize_from_agents
from .analyzer_functions import perplexity_retrieval
from .myvideoindexer import get_image_blob_url, get_uploaded_frames
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import datetime
from django.conf import settings
import numpy as np

class VideoEntityViewSet(viewsets.ModelViewSet):
    queryset = VideoEntity.objects.all()
    serializer_class = VideoEntitySerializer

    def list(self, request, *args, **kwargs):
        account_id = request.query_params.get("account_id")
        if account_id:
            videos = VideoEntity.objects.filter(account_id=account_id)
        else:
            videos = VideoEntity.objects.none()
        print([video.id for video in videos])
        serializer = self.get_serializer(videos, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
        
    def retrieve(self, request, *args, **kwargs):
        account_id = request.query_params.get("account_id")
        instance = self.get_object()

        if not account_id:
            raise PermissionDenied("account_id must be provided in request parameters.")

        if str(instance.account_id) != str(account_id):
            print(f"{account_id} Not Found.")
            return Response({}, status=status.HTTP_200_OK)

        serializer = self.get_serializer(instance)
        return Response(serializer.data, status=status.HTTP_200_OK)
        
    def create(self, request, *args, **kwargs):
        account_id = request.data.get("account_id")
        # Create a new instance and use custom create_video logic
        video = VideoEntity()
        video.create_video(account_id=account_id)
        serializer = self.get_serializer(video)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def update(self, request, *args, **kwargs):
        # Find the instance to update
        video = self.get_object()
        print(request.data)
        update_data = request.data
        video.update_video(**update_data)
        serializer = self.get_serializer(video)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def destroy(self, request, *args, **kwargs):
        video = self.get_object()
        video.delete_video()
        return Response(status=status.HTTP_204_NO_CONTENT)
        
class VideoUploadAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        file_obj = request.FILES.get('file')
        account_id = request.data.get('account_id')
        np_uint64 = np.uint64(int(account_id))
        if not file_obj or not account_id:
            return Response({'error': 'file and account_id are required'}, status=status.HTTP_400_BAD_REQUEST)

        account_name = settings.ACCOUNT_NAME
        container_name = settings.CONTAINER_NAME
        folder = account_id
        blob_name = f"{folder}/{file_obj.name}"

        try:
            blob_service_client = BlobServiceClient(
                account_url=f'https://{account_name}.blob.core.windows.net',
                credential=settings.AZURE_ACCOUNT_KEY
            )
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            blob_client.upload_blob(file_obj, overwrite=True)

            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=container_name,
                blob_name=blob_name,
                account_key=settings.AZURE_ACCOUNT_KEY,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=1)
            )
            sas_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
            # Create a new VideoEntity instance (not saved yet)
            video = VideoEntity()

            # Call the create_video method with required account_id parameter
            video.create_video(account_id=account_id, sas_url=sas_url)
            serializer = VideoEntitySerializer(video)
            # At this point, video is saved with sas_url set to the known constant
            print(f"video with id:{video.id}, account_id: {video.account_id}, sas_url: {video.sas_url} and status:{video.status} created successfully.")

            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def put(self, request, pk = None, format=None):
        sas_url = request.data.get('sas_url')
        video_id = request.data.get('id', pk)
        if not sas_url or not video_id:
            return Response({'error': 'parameter is missing'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Assume video is an existing VideoEntity instance fetched from the database
            video = VideoEntity.objects.get(pk=video_id)

            # video.update_video(sas_url=sas_url)
            update_data = request.data.dict()
            video.update_video(**update_data)
            print(f"video with id:{video.id}, sas_url: {video.sas_url} and status:{video.status} updated successfully.")
            return Response({'sas_url': sas_url, 'id': video.id}, status=status.HTTP_200_OK)
        except VideoEntity.DoesNotExist:
            return Response({"error": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    def delete(self, request, pk = None, format=None):
        video_id = request.data.get('id', pk)
        if not video_id:
            return Response({'error': 'parameter is missing'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            # Assume video is an existing VideoEntity instance fetched from the database
            video = VideoEntity.objects.get(pk=video_id)

            video.delete_video()
            
            print(f"video with id:{video_id} deleted successfully.")
            return Response({'sas_url': sas_url, 'id': video.id}, status=status.HTTP_200_OK)
        except VideoEntity.DoesNotExist:
            return Response({"error": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)        
            
            
class ChatAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def put(self, request, pk=None, format=None):
        account_id = request.data.get('account_id')
        video_id   = request.data.get('video_id')
        query_text = request.data.get('query')
        frames_list = request.data.get("frames")
        np_uint64 = np.uint64(int(account_id))
        if not account_id or not query_text:
            return Response({'error': 'query_text and account_id are required'}, status=status.HTTP_400_BAD_REQUEST)

        account_name = settings.ACCOUNT_NAME
        container_name = settings.CONTAINER_NAME
        folder = account_id
        blob_name = f"{folder}/images"
        if video_id:
           blob_name += f"/{video_id}"
        else:
           video_id = None
           try:
               video_id =  VideoEntity.objects.filter(account_id=account_id).last().id
           except:
               pass
           if not video_id:
              return Response({'text': "Please upload content to analyze.",'imageUrl': None, 'downloadUrl': None}, status=status.HTTP_200_OK)
           blob_name += f"/{video_id}"
        print(f"account={account_name}, container={container_name}, folder={folder}, blob={blob_name}, video_id={video_id}")
        # account=sadronevideo, container=input, folder=2, blob=2/images/1, video_id=1
        try:
            video_sas_url = VideoEntity.objects.get(pk=video_id).sas_url
            # print(f"video_sas_url={video_sas_url}")
            sas_url_template = get_image_blob_url(video_sas_url, 0, folder='images', prefix='frame', include_name=False, video_id=None)
            print(f"sas_url_template={sas_url_template}")
            highest = get_uploaded_frames(video_sas_url, account_id = account_id, video_id = video_id)
            frames = []
            if highest:
                print(f"highest={highest}")
                frames = [str(0), str(int(highest/2)), str(highest-1)]
            if frames_list:
                frames = frames_list.strip(',').split(',')
            print(frames)
            if not frames:
                frames =  [str(num) for num in list(range(20))]
            sas_url_template = sas_url_template.replace("frame0", "frame(number)")
            response_text = synthesize_from_agents(query_text, account_id)
            # response_text = knowledge_base_search(query_text, account_id)
            # response_text = run_function_tools(query_text, account_id)
            print(f"Answer={response_text}")
            # if not response_text:
            #     response_text = perplexity_retrieval(None, query_text, account_id, frames, sas_url_template, pattern="(number)")
            return Response({'text': response_text,'imageUrl': None, 'downloadUrl': None}, status=status.HTTP_200_OK)
        except Exception as e:
            print(e)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)    