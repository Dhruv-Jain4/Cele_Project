from rest_framework.serializers import Serializer
from django.core import validators
from rest_framework import fields
from rest_framework.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_401_UNAUTHORIZED, HTTP_500_INTERNAL_SERVER_ERROR

class ErrorResponseSerializer(Serializer):
    error = fields.CharField(required = True)

    def get_status_code():
        raise NotImplementedError

class BadRequestResponseSerializer(ErrorResponseSerializer):

    def get_status_code():
        return HTTP_400_BAD_REQUEST
    
class UnauthorizedResponseSerializer(ErrorResponseSerializer):

    def get_status_code():
        return HTTP_401_UNAUTHORIZED

class InternalServerErrorResponseSerializer(ErrorResponseSerializer):
    def get_status_code():
        return HTTP_500_INTERNAL_SERVER_ERROR
