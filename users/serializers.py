from rest_framework.serializers import Serializer
from django.core import validators
from rest_framework import fields
from rest_framework.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_401_UNAUTHORIZED, HTTP_500_INTERNAL_SERVER_ERROR


class RegisterRequestSerializer(Serializer):
    email_id = fields.EmailField(required = True)
    password = fields.CharField(required = True)
    username = fields.CharField(required = True)

class VerifyRequestSerizlizer(Serializer):
    code = fields.IntegerField(required = True)


