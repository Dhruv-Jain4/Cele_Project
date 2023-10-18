from django.conf import settings
import numpy as np
from django.core.mail import send_mail
import datetime
from datetime import timedelta
VERIFICATION_CODE_LIFE = 36000 #(in seconds)
def get_code(length = 4):
    values = "0123456789"
    otp = ""
    for i in range(length):
        otp += np.random.choice(list(values))
    return otp
def send_verification_code(email, username):
    code = get_code()
    subject = "Verify your wakework account"
    message = f"{code} is your verification code to signup to your wakework account .\nThis verification code will expire in {VERIFICATION_CODE_LIFE} seconds"
    from_email = settings.EMAIL_HOST_USER
    to_email = [email]
    try:
        send_mail(subject=subject, message=message, from_email=from_email, recipient_list=to_email, fail_silently=False)
    except Exception as exc:
        print(exc, type(exc))
        raise type(exc)
    return code

def send_verification_code_reset(email, username):
    code = get_code()
    subject = "Verification password for password reset."
    message = f"{code} is your verification code to reset the password of your wakeword account. \nThis verification code will expire in {VERIFICATION_CODE_LIFE} seconds"
    from_email = settings.EMAIL_HOST_USER
    to_email = [email]
    try:
        send_mail(subject=subject, message=message, from_email=from_email, recipient_list=to_email, fail_silently=False)
    except Exception as exc:
        print(exc, type(exc))
        raise type(exc)
    return code


def get_user_id(length = 8):
    values = "0123456789"
    otp = ""
    for i in range(length):
        otp += np.random.choice(list(values))
    return otp

def get_current_time():
    curr_time = datetime.datetime.now(datetime.timezone.utc)
    return curr_time


def code_verification(user, code):
    if user.code == code:
            # print(user.code_created + timedelta(seconds=VERIFICATION_CODE_LIFE))
            # TODO handle different time zones accordingly
        curr_time = get_current_time()
        if curr_time > user.code_created + timedelta(seconds=VERIFICATION_CODE_LIFE):
            return "Expired"
        else:
            return True
    else:
        return False