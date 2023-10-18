from typing import Any
from django.db.models.query import QuerySet
from django.forms.models import BaseModelForm
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.static import serve
from django.views import View
from django.views.generic.edit import CreateView, DeleteView
from django.views.generic.list import ListView
from django.views.generic.detail import DetailView
import datetime 
import shutil
import pandas as pd
from django.core.files.base import ContentFile
import io
import matplotlib.pyplot as plt
from tensorflow import keras
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from tensorflow.data import Dataset
import tensorflow as tf
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http.response import JsonResponse
from .forms import ActivateDataForm, NameForm
from .models import Project, ActivateData, TrainingJob

from django.urls import reverse_lazy, reverse
import os
from .tasks import test_func
import time
import urllib, base64

from django.conf import settings
from .const import *
from .utils import preprocess_dir, TriggerWordDataset, train_model, get_model
# from background_task import background


# @background
def add_data():
    cnt = 0
    while cnt < 10:

        time.sleep(0.2)
        destination_directory = os.path.join(settings.STATICFILES_DIRS[0])

        # Create the destination directory if it doesn't exist
        os.makedirs(destination_directory, exist_ok=True)
        with open(os.path.join(destination_directory, "new.txt"),"a") as f:
            f.writelines(f"This is line number {cnt}")
            f.writelines("\n")
        cnt += 1


class Home(LoginRequiredMixin, View):
    def get(self, request):
        print(self.request.get_host())
        return render(request, os.path.join("home", "home.html"))

def audio(request, file):
    return serve(request, file, document_root = r"static")


class UploadActivateDataView(LoginRequiredMixin, View):
    def get(self, request, pk):
        try:
            project = Project.objects.get(id = pk, user = self.request.user)
        except:
            return HttpResponse("404 not found")


        # print(form)
        form = ActivateDataForm(initial={"project" : project.id})

        # print(form)
    
        form.fields["project"].queryset = Project.objects.filter(user= self.request.user)
        # form.fields["project"].

        # print(form)    
        return render(request, r"home\create_activate_data.html", context={'form' : form} )
        # except Exception as exc:
            # print(exc)
            # return JsonResponse({"error": "Error object not found"})
        
    def post(self, request, pk):
        
        data = ActivateDataForm(data=request.POST, files=request.FILES)
        project = Project.objects.get(id = pk)
        if data.is_valid():
            new_data = data.save(commit=True)
            # print(reverse("home:project_detail")_
            return redirect(reverse("home:project_detail",args = [pk]),pk = pk)
        else:
            # request.session["add_data_form"] = data.data
            # request.session["flag"] = True
            # return redirect(self.request.path)
            return redirect(reverse(r"home:error",args=[project.id]))
    
@method_decorator(csrf_exempt, name='dispatch')
class RecordAudioUpload(LoginRequiredMixin, View):
    
    def get(self, request, pk):
        try:
            project = Project.objects.get(id = pk, user = self.request.user)
        except:
            return HttpResponse("404 not found")
    
        return render(request, r"home\record_audio.html", context={"proj" : project})
    
    def post(self, request, pk):
        try:
            project = Project.objects.get(id = pk, user = self.request.user)
        except:
            return JsonResponse({"error" : "Unaothorize Access"})
        files = request.FILES
        # print(request.FILES["file"].read())
        files = {
            "file" : ContentFile(request.FILES["file"].read(), name=f"audio_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
        }
        data = ActivateDataForm(data = {"project" :project}, files = files)
        project = Project.objects.get(id = pk)

        if data.is_valid():
            new_data = data.save(commit=True)
            # print(reverse("home:project_detail")_
            return JsonResponse({"message" : "success"}, status = 200)
            
        else:
            # request.session["add_data_form"] = data.data
            # request.session["flag"] = True
            # return redirect(self.request.path)
            return JsonResponse({"message" : "error"},status = 400)

def error(request, pk):    
    return render(request, "home\error.html",context={"id" : pk})

class CreateProject(LoginRequiredMixin,CreateView):
    model = Project
    fields = ["name"]
    template_name = r"home\create_project.html"
    success_url = reverse_lazy("home:list_projects")
    
    def form_valid(self, form: BaseModelForm) -> HttpResponse:
        object = form.save(commit= False)
        object.user = self.request.user
        object.save()
        return super().form_valid(form)
class ListProject(LoginRequiredMixin,ListView):
    model = Project
    context_object_name = "objects"
    template_name = r"home\list_projects.html"

    def get_queryset(self) -> QuerySet[Any]:
        queryset = super().get_queryset()
        queryset = queryset.filter(user = self.request.user)
        return queryset
    
class DetailProjectView(LoginRequiredMixin, DetailView):
    model = Project
    context_object_name = "project"
    template_name = r"home\project_detail.html"

    def get_queryset(self) -> QuerySet[Any]:
        data = super().get_queryset()
        data = data.filter(user = self.request.user)
        return data

class TrainView(LoginRequiredMixin, View):
    def get(self, request, pk):
        try:
            proj = Project.objects.get(id = pk, user = self.request.user)
        except:
            return HttpResponse("404 Not Found")
        if len(proj.activatedata_set.all()) < 10:
            return render(request, r"home\train_error.html", context={"proj" : proj})
        form = NameForm()
        return render(request, r"home\train.html", context={'form' : form})

    def post(self, request, pk):
        try:
            proj = Project.objects.get(id = pk, user = self.request.user)
        except:
            return HttpResponse("404 Not Found")
        
        name = request.POST["name"]
        epochs = request.POST["epochs"]
        tj = TrainingJob(name = name,project = proj, status = "Training", file_name = f"{proj.id}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt")
        tj.save()
        file_name = "static/progress/" + tj.file_name
        with open(file_name, "a") as f:
            f.writelines("<h1>Training Job Start</h1>")
            f.writelines("<h2>")
        url = reverse("home:train_detail", args=[proj.id, tj.id])
        url = self.request.get_host() + url
        test_func.delay(tj.id, int(epochs), url )
        # test_func(tj.id)
        return redirect(reverse("home:train_detail", args=[proj.id, tj.id]))
        # shutil.copy(i.file.name, os.path.join(temp_dir_name, ))




class DeleteTrainingJobView(LoginRequiredMixin, View):
    def get(self, request, pk, pk2):
        try:
            proj = Project.objects.get(id = pk, user = self.request.user)
        except:
            return HttpResponse("404 Not Found")
        try:
            tj = TrainingJob.objects.get(id = pk2, project = proj)
        except:
            return HttpResponse("404 not Found")
        return render(request, r"home\delete_tr_job.html", context={"name" : tj.name})
    
    def post(self, request, pk, pk2):
        try:
            proj = Project.objects.get(id = pk, user = self.request.user)
        except:
            return HttpResponse("404 Not Found")
        try:
            tj = TrainingJob.objects.get(id = pk2, project = proj)
        except:
            return HttpResponse("404 not Found")
        name = tj.file_name[:-4]
        try:
            os.remove(os.path.join("static", "progress",  tj.file_name))
        except:
            print("Not deleted progress file")
            pass
        try:
            shutil.rmtree(os.path.join("static", "models",  name))
        except:
            print("Not deleted model")
            pass    
        tj.delete()
        return redirect(reverse("home:project_detail", args=[proj.id]))


class TrainDetailsView(LoginRequiredMixin, View):
    def get(self, request, pk, pk2):
        try:
            proj = Project.objects.get(id = pk, user = self.request.user)
        except:
            return HttpResponse("404 Not Found")
        try:
            tj = TrainingJob.objects.get(id = pk2, project = proj)
        except:
            return HttpResponse("404 not Found")
        file_name = os.path.join(r"static\progress", tj.file_name)
        def create_plot(df, tj, col):
            if df is not None:
                tj.file_name
                
                # print(df)
                plt.figure(figsize=(8, 6))
                
                plt.plot(df["epoch"], df[col], )
                plt.xlabel("Epochs", fontdict={"size" : 15})
                plt.ylabel(col.capitalize(), fontdict={"size" : 15})
                plt.title(col.capitalize(), fontdict={"size" : 25, "color" : "darkblue"})  
                # Convert plot to PNG image
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                
                buf.seek(0)
                string = base64.b64encode(buf.read())

                uri = urllib.parse.quote(string)
            else:
                fig = plt.figure(figsize=(8, 6))
                plt.xlabel("Epochs", fontdict={"size" : 15})
                plt.ylabel(col.capitalize(), fontdict={"size" : 15})
                plt.title(col.capitalize(), fontdict={"size" : 25, "color" : "darkblue"})  
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                
                buf.seek(0)
                string = base64.b64encode(buf.read())

                uri = urllib.parse.quote(string)
            return uri
        
        with open(file_name, "r") as f:
            text = f.read()
        try:
            
            df = pd.read_csv(os.path.join("static", "models",tj.file_name[:-4], "training_logs.csv"))
            
            uri1 = create_plot(df, tj, "loss")
            uri2 = create_plot(df, tj, "accuracy")
            # print(uri2)
            # print(uri1)
        except:
            uri1 = None
            uri2 = None
        most_recent_file = None
        most_recent_time = 0
        
        if tj.status == "Completed":
            for entry in os.scandir(os.path.join(r"static/models", tj.file_name[:-4])):
                if entry.is_file():
        # get the modification time of the file using entry.stat().st_mtime_ns
                    mod_time = entry.stat().st_mtime_ns
                    if mod_time > most_recent_time:
                        # update the most recent file and its modification time
                        most_recent_file = entry.name
                        most_recent_time = mod_time
            file_name = os.path.join("models", tj.file_name[:-4], most_recent_file)
        else:
            file_name = None

        return render(request, r"home\train_details.html", context={"text" : text.replace("\n", "<br>"), "proj" : proj, "tj" : tj, "plot" : uri1, "plot2" : uri2, "file" : file_name})
        


class DeleteActivateDataView(LoginRequiredMixin, View):
    def get(self, request, pk, pk2):
        try:
            proj = Project.objects.get(id = pk, user = self.request.user)
        except:
            return HttpResponse("404 Not Found")
        
        try:
            file = ActivateData.objects.get(id = pk2, project = proj)
        except:
            return HttpResponse("404 Not Found")
        
        return render(request, r"home\delete_activate_data.html", context={"file" : file})
    
    def post(self, request, pk, pk2):
        try:
            proj = Project.objects.get(id = pk, user = self.request.user)
        except:
            return HttpResponse("404 Not Found")
        
        try:
            file = ActivateData.objects.get(id = pk2, project = proj)
        except:
            return HttpResponse("404 Not Found")
        
        file.delete()
        return redirect(reverse("home:project_detail", args=[proj.id]), context={'project' : proj})



    