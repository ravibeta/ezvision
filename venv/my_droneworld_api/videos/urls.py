from rest_framework.routers import DefaultRouter
from django.urls import path
from .views import VideoEntityViewSet, VideoUploadAPIView, ChatAPIView

router = DefaultRouter()
router.register(r'video-entities', VideoEntityViewSet)

urlpatterns = router.urls + [
    path('upload-video/', VideoUploadAPIView.as_view(), name='upload-video'),
    path('chat/', ChatAPIView.as_view(), name='chat')
]
