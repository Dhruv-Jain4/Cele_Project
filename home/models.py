from django.db import models
from django.core.exceptions import ValidationError
from users.models import PlatformUser
import os
# Create your models here.
class Project(models.Model):
    name = models.CharField(max_length=128, null=False)
    user = models.ForeignKey(PlatformUser, on_delete=models.CASCADE, null=False)

    def __str__(self) -> str:
        return self.name
# def get_upload_path(instance, filename):
#     # print("hello came here", instance.project.user)

#     return os.path.join("static",instance.project.user.username, filename)
def validate_file_extension(value):
    print("came here")
    if not value.name.endswith('.wav'):
        raise ValidationError('Only wav files are allowed')
class ActivateData(models.Model):
    file = models.FileField("File (only .wav format)",upload_to="static", )
    # user = models.ForeignKey(PlatformUser, on_delete=models.CASCADE, null=False)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, null=False)


class TrainingJob(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, null=False)
    name = models.CharField(max_length=128, null= True)
    start_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(null=True)
    file_name = models.CharField(max_length=512)
    status = models.CharField(max_length=64, choices=[("training", "Training"), ("completed", "Completed"), ("failed", "Failed")])
    
    def __str__(self) -> str:
        return self.name