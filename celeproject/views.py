from django.shortcuts import render
import os
from django.views.static import serve


def no_login_home(request):
    return render(request, r"accounts\no_login_page.html")


