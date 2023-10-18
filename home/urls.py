from django.shortcuts import render
from django.urls import path, include, re_path
from .views import (Home, audio, UploadActivateDataView, CreateProject,
                     ListProject, DetailProjectView, TrainView, DeleteActivateDataView, TrainDetailsView, error, DeleteTrainingJobView, RecordAudioUpload
                     )

from django.conf.urls.static import static
# Create your views here.
app_name = "home"
urlpatterns = [
    path("", Home.as_view(), name = "home"),
    path("project/static/<str:file>", audio , name = "audio"),
    path("project/<int:pk>/uploadfile", UploadActivateDataView.as_view() , name = "upload_audio"),
    path("project/<int:pk>/record_file", RecordAudioUpload.as_view() , name = "record_audio"),
    path("project/<int:pk>/audio/<int:pk2>", DeleteActivateDataView.as_view() , name = "delete_audio"),
    path("project/create", CreateProject.as_view() , name = "create_project"),
    path("project/list", ListProject.as_view() , name = "list_projects"),
    path("project/<int:pk>", DetailProjectView.as_view() , name = "project_detail"),
    path("project/<int:pk>/train", TrainView.as_view(), name="train"),
    path("project/<int:pk>/train/<int:pk2>", TrainDetailsView.as_view(), name="train_detail"),
    path("project/<int:pk>/train/delete/<int:pk2>", DeleteTrainingJobView.as_view(), name="train_delete"),
    path("error/<int:pk>", error, name="error")
    # path("project/train", TrainView.as_view(), name="train"),
]