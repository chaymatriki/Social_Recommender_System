from django.urls import path
from .views import   addUser, edit, dashboard, insert_data,gestionEmp,delete, dashboardEmp,editEmp, loginPage
from django.urls import reverse_lazy
from django.contrib.auth.views import (LoginView, LogoutView, PasswordResetDoneView, PasswordResetView,
                                       PasswordResetCompleteView, PasswordResetConfirmView,
                                       PasswordChangeView, PasswordChangeDoneView,
                                       PasswordResetDoneView)

app_name = 'authapp'

urlpatterns = [
    #path('register/', register, name='register'),
    #******gestionEmp***********
    path('gestionEmp/', gestionEmp, name='gestionEmp'),
    #**********edit_Admin*****************************
    path('edit/', edit, name='edit'),
    #********dash Admin *******************************
    path('dashboard/', dashboard, name='dashboard'),
    #********dash Emp **********************************
    path('dashboardEmp/', dashboardEmp, name='dashboardEmp'),
    #*****************EDIT PROFILE EMP *****************
    path('editEmp/', editEmp, name='editEmp'),
    #*****************************LOGIN*****************
    path('login/', loginPage, name='login'),
    #path('login/', LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', LogoutView.as_view(template_name='authapp/logged_out.html'), name='logout'),
    path('password_change/', PasswordChangeView.as_view(
        template_name='authapp/password_change_form.html'), name='password_change'),
    path('password_change/dond/', PasswordChangeDoneView.as_view(template_name='authapp/password_change_done.html'),
         name='password_change_done'),
    #**************Envoie de mail :mdp forgotten ************************
    path('password_reset/', PasswordResetView.as_view(
        template_name='authapp/password_reset_form.html',
        email_template_name='authapp/password_reset_email.html',
        success_url=reverse_lazy('authapp:password_reset_done')), name='password_reset'),
    path('password_reset/done/', PasswordResetDoneView.as_view(
        template_name='authapp/password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', PasswordResetConfirmView.as_view(
        template_name='authapp/password_reset_confirm.html',
        success_url=reverse_lazy('authapp:login')), name='password_reset_confirm'),
    path('reset/done/', PasswordResetCompleteView.as_view(
        template_name='authapp/password_reset_complete.html'), name='password_reset_complete'),
    #********DELETE***************************************
    path('delete/<int:emp_id>',delete, name='delete'),
    #******UPDATE*****************************************
    #path('updateEmp/<int:emp_id>/',updateEmp),
    #path('update_data/<int:emp_id>',update_data, name='update_data'),
    #********AJOUT***********************************
    path('addUser/', addUser),
    path('insert_data/', insert_data, name='insert_data'),


]
