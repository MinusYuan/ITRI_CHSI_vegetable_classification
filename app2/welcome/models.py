from django.db import models
from django.core.files.storage import FileSystemStorage
from django.conf import settings

# Create your models here.
class log_history(models.Model):
    filename = models.CharField(max_length=50)
    written_time = models.DateTimeField()
