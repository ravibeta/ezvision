import logging
import sys
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import VideoEntity
from .myvideoindexer import indexing_workflow
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@receiver(post_save, sender=VideoEntity)
def video_post_save_handler(sender, instance, created, **kwargs):
    """
    Signal handler that runs after a VideoEntity instance is saved.
    Calls the notify function.
    """
    try:
        # notify(instance, created=created)  # Pass instance and created flag if useful
        video_url = indexing_workflow(instance.sas_url, instance.account_id, instance.id)
        if video_url:
            logger.info(f"Indexed video now available at {video_url}")
        else:
            logger.info(f"Uploaded video could not be indexed. Please try again later.")
    except Exception as e:
        # Optional: log the error if notify fails
        logger.info(f"Error calling notify for VideoEntity {instance.id}: {e}")
        # import logging
        # logger.error(f"Error calling notify for VideoEntity {instance.id}: {e}")
        # logger = logging.getLogger(__name__)
