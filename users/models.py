from django.db import models
from django.contrib.auth.models import AbstractUser
# Create your models here.
class   PlatformUser(AbstractUser):
    user_id = models.IntegerField(blank=True, null=True)
    is_verified = models.IntegerField(default=False)
    code = models.CharField(max_length=32, null= True)
    code_created = models.DateTimeField(auto_now=True)
    email = models.EmailField(blank = False, unique= True)
    username = models.CharField(blank=False, unique=True, max_length=128)

class ResetPassword(models.Model):
    user = models.ForeignKey(PlatformUser, on_delete=models.CASCADE)
    code = models.CharField(max_length=32)
    code_created = models.DateTimeField()
    code_verified = models.BooleanField()
    verified_at = models.DateTimeField(null=True)