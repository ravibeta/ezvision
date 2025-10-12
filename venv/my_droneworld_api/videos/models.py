from django.db import models
from urllib.parse import urlparse

class VideoEntity(models.Model):
    class Status(models.TextChoices):
        INITIALIZED = "Initialized", "Initialized"
        PROCESSING = "Processing", "Processing"
        COMPLETED = "Completed", "Completed"
        CANCELED = "Canceled", "Canceled"
        RESERVED = "Reserved", "Reserved"
    account_id = models.CharField(max_length=255)
    video_url = models.CharField(null=True, blank=True, max_length=1024)
    index_name = models.CharField(null=True, blank=True, max_length=255)
    sas_url = models.URLField(max_length=500)
    file_name = models.CharField(null=True, blank=True, max_length=255)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.INITIALIZED,
    )
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"VideoEntity(id={self.id}, account_id={self.account_id}, status={self.status})"
        
    def create_video(self, account_id, sas_url = None):
        """
        Sets the sas_url to a known constant and saves the new instance.
        """
        self.account_id = account_id
        self.sas_url = sas_url
        self.file_name = self.get_name_from_url(sas_url)
        self.status = self.Status.INITIALIZED
        self.save()

    def update_video(self, **kwargs):
        """
        Updates instance fields with provided kwargs and saves.
        """
        for field, value in kwargs.items():
            if hasattr(self, field):
                setattr(self, field, value)
        self.status = self.Status.INITIALIZED
        self.save()
        #        # VideoEntity.objects.filter(pk=self.pk).update(field=value)
        #VideoEntity.objects.filter(pk=self.pk).update(status=Status.INITIALIZED)
        

    def delete_video(self):
        """
        Deletes the current instance from the database.
        """
        self.delete()
        
    def get_name_from_url(self, sas_url):
        parsed = urlparse(sas_url)
        path_parts = parsed.path.split('/')
        blob_name = path_parts[-1]
        return blob_name
        
class ImageEntity(models.Model):
    video = models.ForeignKey(
        VideoEntity,
        related_name="images",
        on_delete=models.CASCADE
    )
    account_id = models.CharField(max_length=255)
    video_url = models.CharField(null=True, blank=True, max_length=1024)
    index_name = models.CharField(max_length=255)
    sas_url = models.CharField(max_length=1024)
    description = models.TextField(null=True, blank=True, max_length=4096)
    timestamp  = models.TimeField()
    location = models.CharField(max_length=255)
    status = models.CharField(max_length=255)
    created = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"ImageEntity(id={self.id}, video_id={self.video_id})"