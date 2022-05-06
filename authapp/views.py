from django.contrib import messages
from django.shortcuts import redirect, render
from django.contrib.auth.decorators import login_required
from .forms import  EmpEditForm, UserRegistration, UserEditForm
from .models import Empl
from .forms import Add
from django.contrib.auth import authenticate,login
from django.http import *
from django.contrib.auth.models import User
from .decorators import  admin_only, allowed_users
from django.contrib.auth.models import Group
# Create your views here.
#**********************LOGIN********************
def loginPage(request):

	if request.method == 'POST':
		username = request.POST.get('username')
		password =request.POST.get('password')

		user = authenticate(request, username=username, password=password)

		if user is not None:
			login(request, user)
			return redirect('authapp:dashboard')
		else:
			messages.info(request, 'Your username and password didn\'t match! Please try again.')

	context = {}
	return render(request, 'registration/login.html', context)

@login_required(login_url='login')
@admin_only
#***************DASH ADMIN*******************
def dashboard(request):
    return render(request, 'authapp/dashboard.html')
    
#def register(request):
    
 #   if request.method == 'POST':
  #      form = UserRegistration(request.POST or None)
   #     if form.is_valid():
    #        new_user = form.save(commit=False)
     #       new_user.set_password(
      #          form.cleaned_data.get('password')
       #     )
        #    new_user.save()
         #   return render(request, 'authapp/register_done.html')
    #else:
     #   form = UserRegistration()

#means inforamation that template needs
    #context = {
     #   "form": form
    #}

    #return render(request, 'authapp/register.html', context=context)

@login_required
@allowed_users(allowed_roles=['admin'])
#******************Edit profile Admin*******************

def edit(request):
    admin = request.user.admin
    form =  UserEditForm(instance=admin)
    if request.method == 'POST':
        form = UserEditForm(request.POST, request.FILES,instance=admin)
        if form.is_valid():
            form.save()
    else:
        form = UserEditForm(instance=admin)
    context = {
        'form': form,
    }
    return render(request, 'authapp/edit.html', context=context)

@login_required
@allowed_users(allowed_roles=['admin'])
#***************Gestion des employées*****************
def gestionEmp(request):
    employees = Empl.objects.all()
    return render(request, 'authapp/gestionEmp.html',{'employees': employees})

@login_required
@allowed_users(allowed_roles=['admin'])
#*******************DELETE EMPLOYEE******************************
def delete(request, emp_id):
    
    employees = Empl.objects.all()
    emp = Empl.objects.get(id=emp_id)    
    emp.delete()
    return render(request, 'authapp/gestionEmp.html',  {'employees': employees})

@login_required
@allowed_users(allowed_roles=['admin'])
#****************AJOUT EMPLOYEE********************************************
def addUser(request):
    form = Add()
    return render(request, 'authapp/addUser.html', {'form': form}) 
def insert_data(request):
    #employees = Employee.objects.all()
    employees = Empl.objects.all()
    users = User.objects.all()
    if request.method == 'POST' :
        username = request.POST['username']
        fname = request.POST['surname']
        lname = request.POST['name']
        email = request.POST['email']
        entreprise = request.POST['entreprise']
        mdp = request.POST['password']
        #new = Employee(username=username, first_name=fname, last_name=lname,email=email,password=mdp,entreprise=entreprise)
       
        #user = User(username=username, first_name=fname, last_name=lname,email=email,password=mdp)
        user = User(username=username,password=mdp)
        user.save()
        new = Empl(user=user, fname=fname, lname=lname,email1=email,entreprise=entreprise)
        new.save()  
        #********nécessaire pour mettre l'employee ajouté ds le groupe "employee"******
        group= Group.objects.get(name='employee')
        user.groups.add(group)      
        return render(request, 'authapp/gestionEmp.html',  {'employees': employees})
    else:
        render(request, "authapp/gestionEmp.html")

@login_required
@allowed_users(allowed_roles=['employee'])

#*****************DASH EMPLOYEE ***************************
def dashboardEmp(request):
    return render(request, 'authapp/dashboardEmp.html')

@login_required
@allowed_users(allowed_roles=['employee'])
#******************Edit profile Employee*******************
def editEmp(request):
    emp = request.user.empl
    form = EmpEditForm(instance=emp)
    if request.method == 'POST':
        form = EmpEditForm(request.POST, request.FILES,instance=emp)
        if form.is_valid():
            form.save()
    else:
        form = EmpEditForm(instance=emp)
    context = {
        'form': form,
    }
    return render(request, 'authapp/profilEmp.html', context=context)

