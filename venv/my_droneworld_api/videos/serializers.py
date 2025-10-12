from rest_framework import serializers
from .models import VideoEntity

class VideoEntitySerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoEntity
        fields = '__all__'
