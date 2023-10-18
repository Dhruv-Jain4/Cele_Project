# forms.py
from typing import Any
from django.forms import fields
from django import forms
from rest_framework.fields import SerializerMethodField
from .serializers import RegisterRequestSerializer
from django.conf import settings
import logging
from django.contrib.auth import get_user_model
logging.basicConfig(format=f"%(pathname)s - %(lineno)s - %(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s)",filename="logs.txt",filemode="a",level=logging.INFO)

LOGGER = logging.getLogger(__name__)

def user_exists(**kwargs):
    user_model = get_user_model()
    try:
        user = user_model.objects.get(**kwargs)
        return user
    except Exception as exc:
        LOGGER.info(f"User with given details does not exist {str(kwargs)}")
        return False


class RegisterForm(forms.Form):
    email_id = fields.EmailField(required = True)
    password = forms.CharField(widget=forms.PasswordInput)
    username = fields.CharField(required = True)

    def clean(self) -> dict[str, Any]:

        data =  super().clean()
        # print(self.data
        mail = data.get('email_id', '')
        username = data.get("username", "")
        user= user_exists(**{"email" : mail})
        if user:
            # user = user_model.objects.get(email = mail)
            if user.is_verified:
                self.add_error("email_id","Email id already taken")
            else:
                if username == user.username:
                    return
        else:
            pass
        
        user= user_exists(**{"username" : username})
        if user:
            self.add_error("username","username already taken")
        else:
            pass

        if len(username) < 7:
            self.add_error("username","Username must be of atleast 7 characters")
        
class VerifyForm(forms.Form):
    code = fields.IntegerField()


class ForgotPasswordForm(forms.Form):
    email=fields.EmailField()

    def clean(self) -> dict[str, Any]:
        data = super().clean()
        mail = data.get("email")

        user= user_exists(**{"email" : mail})
        if user:
            pass
        else:
            self.add_error("email", "Given email id is not a registered email id")