from django.shortcuts import render, redirect
from django.urls import reverse, reverse_lazy
# from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.views import View
from drf_yasg.utils import swagger_auto_schema
from .serializers import RegisterRequestSerializer
from .forms import RegisterForm, VerifyForm, ForgotPasswordForm
from django.http.response import HttpResponse
from utils.serializers import BadRequestResponseSerializer, UnauthorizedResponseSerializer, InternalServerErrorResponseSerializer
import os
import json
from utils.utils import send_verification_code, get_user_id, code_verification, send_verification_code_reset
from .models import PlatformUser, ResetPassword
import datetime
from django.contrib.auth.views import LoginView, PasswordResetView, PasswordChangeDoneView, PasswordResetConfirmView, PasswordResetCompleteView, LogoutView, PasswordResetDoneView, PasswordChangeView
# Create your views here.
import logging
import datetime
from django.views.static import serve

logging.basicConfig(format=f"%(pathname)s - %(lineno)s - %(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s)",filename="logs.txt",filemode="a",level=logging.INFO)

LOGGER = logging.getLogger(__name__)

class RegisterView(View):
    @swagger_auto_schema(request_body=RegisterRequestSerializer)
    def post(self, request):
        form = RegisterForm(data=request.POST)
        if form.is_valid():
            user, created = PlatformUser.objects.get_or_create(email = form.data["email_id"], defaults={"username" :form.data["username"]})
            # print(form.data["password"])
            if not created:
                user.username = form.data["username"]
            user.set_password(form.data["password"])
            user.is_verified = False
            user.user_id = get_user_id()
            try:
                print(user.email, user.password)
                code = send_verification_code(user.email, user.username)
            except Exception as exc:
                LOGGER.info(f"Unable to send mail due to the error {exc}")
                return HttpResponse("Unable to send mail to given mail id as it does not exist")
            user.code = code
            user.code_created = datetime.datetime.now()
            user.save()

            LOGGER.info(f"Verification mail sent to the mail {user.email}")
            request.session["mail"] = user.email
            return redirect(reverse("platform_users:verify"))
        else:
            LOGGER.info("Invalid register form data")
            request.session["register_form"] = form.data
            # request.session["errors"] = form.errors
            return redirect(request.path)

    def get(self, request):
        form = request.session.get("register_form",None)
        # print(form)
        if not form:
            form = RegisterForm()
        else:
            form = RegisterForm(form)

        try:
            del request.session["register_form"]
        except:
            pass
        context  ={
            "form" : form
        }
        # print(form)
        return render(request, os.path.join("accounts", "register.html"),context=context)



class VerifyView(View):
    def get(self, request):
        form = request.session.get("verify_form",None)
        try:
            mail = request.session["mail"]
        except:
            return HttpResponse("401 Unauthorized", status_code = 401)

        # print(form)
        if not form:
            form = VerifyForm()
        else:
            form = VerifyForm(form)
            form.add_error("code", "Invalid code")
        try:
            del request.session["verify_form"]
        except:
            pass
        context  ={
            "form" : form
        }
        # print(form)
        return render(request, os.path.join("accounts", "verify.html"),context=context)
    
    def post(self, request):
        form = VerifyForm(data=request.POST)
        try:
            mail = request.session["mail"]
        except:
            return HttpResponse("401 Unauthorized", status_code = 401)
        try:
            user = PlatformUser.objects.get(email = mail)
        except:
            return HttpResponse("401 Unauthorized access", status_code = 401)

        if form.is_valid():
            verification_status = code_verification(user, form.data["code"])
            if verification_status:
                if verification_status == "Expired":
                    try:
                        del request.session["mail"]
                    except:
                        pass
                    return render(request, os.path.join("accounts", "code_expired.html"))
                user.is_verified = True
                user.save()
                try:
                    del request.session["mail"]
                except:
                    pass
                return redirect(reverse("platform_users:login"))
            else:
                request.session["verify_form"] = form.data
            # request.session["errors"] = form.errors
                return redirect(request.path)    
        else:
            request.session["verify_form"] = form.data
            # request.session["errors"] = form.errors
            return redirect(request.path)




class HomeView(View):
    def get(self, request):
        return render(request, os.path.join("accounts", "home.html"))
    
class MyLoginView(LoginView): ... 

class MyPasswordResetView(PasswordResetView): 
    email_template_name = os.path.join("accounts","password_reset_email.html")
    template_name = os.path.join("accounts","password_reset_form.html")
    success_url = reverse_lazy("platform_users:password_reset_done")

class MyPasswordChangeDoneView(PasswordChangeDoneView): 
    template_name = os.path.join("accounts","password_change_done.html")
    

class MyPasswordResetConfirmView(PasswordResetConfirmView): 
    template_name = os.path.join("accounts", "password_reset_confirm.html")
    success_url = reverse_lazy("platform_users:password_reset_complete")

class MyPasswordResetCompleteView(PasswordResetCompleteView): 
    template_name = os.path.join("accounts", "password_reset_complete.html")

class MyLogoutView(LogoutView): 
    template_name = os.path.join("accounts", "no_login_page.html")

class MyPasswordResetDoneView(PasswordResetDoneView):
    template_name = os.path.join("accounts","password_reset_done.html")
    

class MyPasswordChangeView(PasswordChangeView): 
    template_name = os.path.join("accounts", "password_change_form.html")
    success_url = reverse_lazy("platform_users:password_change_done")
# class ForgotPasswordView(View):
#     def get(self, request):
#         form = request.session.get("forgot_pass_form", None)
#         if not form:
#             form = ForgotPasswordForm()
#         else:
#             form = ForgotPasswordForm(form)
#         try:
#             del request.session["forgot_pass_form"]
#         except:
#             pass
#         context = {
#             "form" : form
#         }

#         return render(request, os.path.join("accounts", "forgot_password.html"), context=context)

#     def post(self, request):
#         form = ForgotPasswordForm(request.POST)
#         if form.is_valid():
#             request.session["mail"] = form.data["email"]
#             request.session["forgot_req"] = True
#             user = PlatformUser.objects.get(email = form.data["email"])
#             code = send_verification_code_reset(user.email, user.username)
#             reset_pass, created = ResetPassword.objects.get_or_create(user = user, code_verified = False,
#                                                                        defaults = {"code" : code, "code_created" : datetime.datetime.now()})
#             reset_pass.code = code
#             reset_pass.code_created = datetime.datetime.now()
#             reset_pass.save()
#             return redirect(reverse("platform_users:forgot_pass_verify"))

#         else:
#             request.session["forgot_pass_form"] = form.data
#             return redirect(request.path)
        
# class ForgotPasswordVerifyView(View):
#     def get(self, request):
#         form = request.session.get("forgot_pass_verify_form",None)
#         try:
#             mail = request.session["mail"]
#             forgot_req = request.session["forgot_req"] = True
#         except:
#             return HttpResponse("401 Unauthorized", status = 401)

#         # print(form)
#         if not form:
#             form = VerifyForm()
#         else:
#             form = VerifyForm(form)
            
#             form.add_error("code", "Invalid code")
#         try:
#             del request.session["forgot_pass_verify_form"]
#         except:
#             pass
#         context  ={
#             "form" : form
#         }
#         # print(form)
#         return render(request, os.path.join("accounts", "verify.html"),context=context)
    
#     def post(self, request):
#         form = VerifyForm(data=request.POST)
#         try:
#             mail = request.session["mail"]
#         except:
#             return HttpResponse("401 Unauthorized", status_code = 401)
#         try:
#             user = PlatformUser.objects.get(email = mail)
#         except:
#             return HttpResponse("401 Unauthorized access", status_code = 401)
#         try:
#             reset_pass = ResetPassword.objects.get(user = user)
#         except:
#             return HttpResponse("401 Unauthorized access", status_code = 401)
#         if form.is_valid():
#             verification_status = code_verification(reset_pass, form.data["code"])
#             if verification_status:
#                 if verification_status == "Expired":
#                     try:
#                         del request.session["mail"]
#                         del request.session["forgot_req"]
#                     except:
#                         pass
#                     return render(request, os.path.join("accounts", "code_expired.html"))
#                 reset_pass.code_verified = True
#                 reset_pass.verified_at = datetime.datetime.now()
#                 reset_pass.save()
#                 try:
#                     del request.session["mail"]
#                     del request.session["forgot_req"]
#                 except:
#                     pass
#                 request.session["mail"] = user.email
#                 request.session["reset_pass"] = True
#                 return redirect(reverse("platform_users:reset_password"))
#             else:
#                 request.session["forgot_pass_verify_form"] = form.data
#             # request.session["errors"] = form.errors
#                 return redirect(request.path)    
#         else:
#             request.session["forgot_pass_verify_form"] = form.data
#             # request.session["errors"] = form.errors
#             return redirect(request.path)



            






