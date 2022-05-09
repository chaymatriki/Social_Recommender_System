from django.contrib import admin
from .models import  Administration,Empl,Admin,Dataset

# Register your models here.
admin.site.register(Administration)
#admin.site.register(Employee)
admin.site.register(Empl)
admin.site.register(Admin)
admin.site.register(Dataset)