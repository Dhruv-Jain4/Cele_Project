from django.contrib import admin
from .models import Project, ActivateData, TrainingJob
# Register your models here.
admin.site.register(Project)
admin.site.register(ActivateData)
admin.site.register(TrainingJob)