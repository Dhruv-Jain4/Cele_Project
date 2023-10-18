from typing import Any
from django.forms import ModelForm, ClearableFileInput, Form, fields
# from django.forms import fields

from .models import ActivateData

class ActivateDataForm(ModelForm):
    class Meta:
        model = ActivateData
        fields = "__all__"

        
    def is_valid(self) -> bool:
        bl = super().is_valid()
        # print(self.files)
        if not self.files["file"].name.endswith(".wav"):
            return False
        else:
            return bl

class NameForm(Form):
    name = fields.CharField(max_length=128, required=True)
    epochs = fields.IntegerField(min_value=2, max_value=100, required=True)