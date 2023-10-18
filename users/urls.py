from django.urls import path, include, re_path
from .views import (RegisterView, VerifyView, HomeView, #ForgotPasswordView,
                MyLoginView, MyLogoutView, MyPasswordChangeDoneView, MyPasswordResetCompleteView,
                MyPasswordResetConfirmView, MyPasswordChangeView, MyPasswordResetDoneView, MyPasswordResetView
                )

app_name = "platform_users"

urlpatterns = [
    path("register/", RegisterView.as_view(), name = "register"),
    path("verify/",VerifyView.as_view(), name = "verify"),
    
    path(
        "reset/<uidb64>/<token>/",
        MyPasswordResetConfirmView.as_view(),
        name="password_reset_confirm",
    ),
    # path("forgotpassworverify",ForgotPasswordVerifyView.as_view(), name = "forgot_pass_verify"),
    
    path("login/", MyLoginView.as_view(), name="login"),
    path("logout/", MyLogoutView.as_view(), name="logout"),
    path(
        "password_change/", MyPasswordChangeView.as_view(), name="password_change"
    ),
    path(
        "password_change/done/",
        MyPasswordChangeDoneView.as_view(),
        name="password_change_done",
    ),
    path("password_reset/", MyPasswordResetView.as_view(), name="password_reset"),
    path(
        "password_reset/done/",
        MyPasswordResetDoneView.as_view(),
        name="password_reset_done",
    ),
    # path(
    #     "reset/<uidb64>/<token>/",
    #     MyPasswordResetConfirmView.as_view(),
    #     name="password_reset_confirm",
    # ),
    
    path(
        "reset/done/",
        MyPasswordResetCompleteView.as_view(),
        name="password_reset_complete",
    ),
    # path("forgotpassword", ForgotPasswordView.as_view(), name="forgotpassword"),
]
