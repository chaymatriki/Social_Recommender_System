from django.db import models
from django.conf import settings
from django.contrib.auth.models import User

# Create your models here.

class Administration(models.Model):
    username = models.CharField(max_length=250)
    first_name = models.CharField(max_length=250)
    last_name = models.CharField(max_length=250)
    email= models.CharField(max_length=250)
    password= models.CharField(max_length=250)
    num_admin= models.IntegerField( default='0')
    entreprise=  models.CharField(max_length=250)

#class Employee(models.Model):
 #   username = models.CharField(max_length=250)
  #  first_name = models.CharField(max_length=250)
   # last_name = models.CharField(max_length=250)
    #email= models.CharField(max_length=250)
    #password= models.CharField(max_length=250)
    #num_admin= models.IntegerField( default='1')
    #entreprise=  models.CharField(max_length=250)

class Admin(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    fname = models.CharField(max_length=200, null=True)
    lname = models.CharField(max_length=200, null=True)
    email1 = models.CharField(max_length=200, null=True)
    num_admin= models.IntegerField( default='0')
    entreprise=  models.CharField(max_length=250)
    profile_pic = models.ImageField(default="2.png", null=True, blank=True)


class Empl(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    fname = models.CharField(max_length=200, null=True)
    lname = models.CharField(max_length=200, null=True)
    email1 = models.CharField(max_length=200, null=True)
    num_admin= models.IntegerField( default='1')
    entreprise=  models.CharField(max_length=250)
    profile_pic = models.ImageField(default="2.png", null=True, blank=True)

class UserRegistrationModel(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

