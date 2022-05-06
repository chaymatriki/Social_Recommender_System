from django.contrib.auth.models import User
from django import forms
# from .models import Profile
from django.contrib.auth.forms import AuthenticationForm, UsernameField
from .models import  Admin, Administration, Empl, UserRegistrationModel
from django.contrib.auth.forms import PasswordResetForm


class UserRegistration(forms.ModelForm):
    password = forms.CharField(label='Password', widget=forms.PasswordInput,)
    password2 = forms.CharField(
        label='Repeat Password', widget=forms.PasswordInput)

    class Meta:
        model =  User
        fields = ('username', 'first_name', 'last_name', 'email')
        

        def clean_password2(self):
            cd = self.cleaned_data
            if cd['password'] != cd['password2']:
                raise forms.ValidationError('Passwords don\'t match.')
            return cd['password2']

#*******************EDIT PROFILE ADMIN***********************
class UserEditForm(forms.ModelForm):
    class Meta:
        #model = User
        model = Admin
        fields = ('fname', 'lname', 'email1','profile_pic')
        exclude = ['user']

#*******************EDIT PROFILE EMPLOYEE***********************
class EmpEditForm(forms.ModelForm):
    class Meta:
        model = Empl
        fields = ('fname', 'lname', 'email1','profile_pic')
        exclude = ['user']

#******************AJOUT EMPLOYEE****************************
class Add(forms.Form):
   username = forms.CharField()
   fname = forms.CharField()
   lname = forms.CharField()
   entreprise = forms.CharField()
   password = forms.CharField()
   email1 = forms.CharField()
