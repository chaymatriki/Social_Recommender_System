from django.contrib import messages
from django.shortcuts import redirect, render
from django.contrib.auth.decorators import login_required
from .forms import  EmpEditForm, UserRegistration, UserEditForm
from .models import Empl,Dataset
from .forms import Add
from django.contrib.auth import authenticate,login
from django.http import *
from django.contrib.auth.models import User
from .decorators import  admin_only, allowed_users
from django.contrib.auth.models import Group
from django.http import request
import pandas as pd
import json 
from django.core.files.storage import FileSystemStorage
import time
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import ClusterEnsembles as CE




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
    context = {}
    if request.method == 'GET':
        return render(request, 'authapp/dashboard.html',context)
    if request.method == 'POST':
        doc = request.FILES
        doc_name = doc['myfile']
        fss = FileSystemStorage()
        file = fss.save(doc_name.name, doc_name)
        uploaded_file_url = fss.url(file)

        
        df = pd.read_csv(uploaded_file_url[1:])
        df = df[:10]
        df.columns = df.columns.str.replace(' ', '')
        column_names = list(df.columns.values)
        json_records = df.reset_index().to_json(orient ='records')
        arr = []
        arr = json.loads(json_records)
  
        print(arr)
        context = {'d': arr , 'c': column_names , 'uploaded_file_url': uploaded_file_url}
        return render(request, 'authapp/dashboard.html',context)
    
@login_required(login_url='login')
@admin_only
def kmeans(request):
    url=request.GET.get('uploaded_file_url')
    df = pd.read_csv(url[1:],index_col=0)
    df.columns = df.columns.str.replace(' ', '')
    
    
    scaler= MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    km_scores = []

    for i in range(2,11):
        km = KMeans(n_clusters=i, random_state=0).fit(df_scaled)
        preds = km.predict(df_scaled)
        km_scores.append(km.inertia_)

    def optimal_number_of_clusters(wcss):
        x1, y1 = 2, wcss[0]
        x2, y2 = 10, wcss[len(wcss)-1]

        distances = []
        for i in range(len(wcss)):
            x0 = i+2
            y0 = wcss[i]
            numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(numerator/denominator)
    
        return distances.index(max(distances)) + 2

    nb_clusters = optimal_number_of_clusters(km_scores)
    kmeans=KMeans(n_clusters=nb_clusters)
    kmeans.fit(df_scaled)

    #metrics
    preds = kmeans.predict(df_scaled)
    silhouette = silhouette_score(df_scaled,preds)
    db = davies_bouldin_score(df_scaled,preds)
    ch = calinski_harabasz_score(df_scaled, preds)

    df['Cluster']=kmeans.labels_

    json_records = df[:10].reset_index().to_json(orient ='records')
    arr = []
    arr = json.loads(json_records)
    
    column_names = list(df.columns.values)
    column_names.insert(0,df.index.name)
    context = {'d': arr , 'c': column_names, 'silhouette': silhouette, 'db': db, 'ch': ch }
    if request.method == 'GET':
        return render(request, 'authapp/kmeans.html',context)
    if request.method == 'POST':
        result_csv = df.to_csv('./datasets/result_'+url[10:], encoding='utf-8')
        dataset = Dataset()
        dataset.dataset_name = url[10:]
        dataset.entreprise_name = 'ent1'
        dataset.dataset_path = 'datasets/result_'+url[10:]
        dataset.save()
        return render(request, 'authapp/kmeans.html')

@login_required(login_url='login')
@admin_only
def pca(request):
    url=request.GET.get('uploaded_file_url')
    df = pd.read_csv(url[1:],index_col=0)
    df.columns = df.columns.str.replace(' ', '')
    
    
    scaler= MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    pca=PCA()
    pca.fit(df_scaled)
    pca_variance = pca.explained_variance_ratio_
    nb_features = len(df.columns)
    nbpc = 0
    varcumsum  = 0.0
    while varcumsum <= 0.8 :
        varcumsum = varcumsum + pca_variance[nbpc]
        nbpc = nbpc + 1
    
    pca=PCA(n_components=nbpc)
    pca.fit(df_scaled)
    scores_pca=pca.transform(df_scaled)
    km_scores= []
    for i in range(2,11):
        km = KMeans(n_clusters=i, random_state=0).fit(scores_pca)
        preds = km.predict(scores_pca)
        km_scores.append(km.inertia_)
    def optimal_number_of_clusters(wcss):
        x1, y1 = 2, wcss[0]
        x2, y2 = 10, wcss[len(wcss)-1]
        distances = []
        for i in range(len(wcss)):
            x0 = i+2
            y0 = wcss[i]
            numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(numerator/denominator)
        return distances.index(max(distances)) + 2
    nb_clusters = optimal_number_of_clusters(km_scores)
    kmeans_pca=KMeans(n_clusters=nb_clusters)
    kmeans_pca.fit(scores_pca)
    
    #metrics
    preds = kmeans_pca.predict(scores_pca)
    silhouette = silhouette_score(scores_pca,preds)
    db = davies_bouldin_score(scores_pca,preds)
    ch = calinski_harabasz_score(scores_pca, preds)

    y_axis = list(abs(pca.components_[0])*100)
    
    x_axis = list(df.columns.values)
    df['Cluster'] = kmeans_pca.labels_
    json_records = df[:10].reset_index().to_json(orient ='records')
    arr = []
    arr = json.loads(json_records)
    column_names = list(df.columns.values)
    column_names.insert(0,df.index.name)
    context = {'d': arr , 'c': column_names, 'x_axis': x_axis, 'y_axis': y_axis, 'silhouette': silhouette, 'db': db, 'ch': ch }
    
    if request.method == 'GET':
        return render(request, 'authapp/pca.html',context)
    if request.method == 'POST':
        result_csv = df.to_csv('./datasets/result_'+url[10:], encoding='utf-8')
        dataset = Dataset()
        dataset.dataset_name = url[10:]
        dataset.entreprise_name = 'ent1'
        dataset.dataset_path = 'datasets/result_'+url[10:]
        dataset.save()
        return render(request, 'authapp/pca.html')







@login_required(login_url='login')
@admin_only
def autoencoder(request):
    url=request.GET.get('uploaded_file_url')
    df = pd.read_csv(url[1:],index_col=0)
    df.columns = df.columns.str.replace(' ', '')   
    scaler= MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    train, test = train_test_split(df_scaled, test_size=0.2)
    nb_features = len(df.columns)
    input_df = Input( shape = (nb_features, ))
    x = Dense(round(nb_features * 0.8), activation = 'relu', kernel_initializer='glorot_uniform')(input_df)
    x = Dense(round(nb_features * 0.6), activation = 'relu', kernel_initializer='glorot_uniform')(input_df)
    x = Dense(round(nb_features * 0.4), activation = 'relu', kernel_initializer='glorot_uniform')(x)
    encoded = Dense(round(nb_features * 0.3), activation = 'relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(round(nb_features * 0.4), activation = 'relu', kernel_initializer='glorot_uniform')(encoded)
    x = Dense(round(nb_features * 0.6), activation = 'relu', kernel_initializer='glorot_uniform')(input_df)
    x = Dense(round(nb_features * 0.8), activation = 'relu', kernel_initializer='glorot_uniform')(x)
    decoded = Dense(nb_features, kernel_initializer='glorot_uniform')(x)
    autoencoder = Model(input_df, decoded)
    encoder = Model(input_df, encoded)
    autoencoder.save('autoencoder.h5')
    autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')
    history = autoencoder.fit(train, train, batch_size= 300, epochs = 35, verbose = 1, validation_data=(test,test))
    pred = encoder.predict(df_scaled)
    score = []
    range_values = range(2, 11)
    for i in range_values:
        kmeans = KMeans(n_clusters = i)
        kmeans.fit(pred)
        score.append(kmeans.inertia_)
    def optimal_number_of_clusters(wcss):
        x1, y1 = 2, wcss[0]
        x2, y2 = 10, wcss[len(wcss)-1]
        distances = []
        for i in range(len(wcss)):
            x0 = i+2
            y0 = wcss[i]
            numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(numerator/denominator)
        return distances.index(max(distances)) + 2
    nb_clusters = optimal_number_of_clusters(score)
    kmeans = KMeans(nb_clusters)
    kmeans.fit(pred)

    #metrics
    preds = kmeans.predict(pred)
    silhouette = silhouette_score(pred,preds)
    db = davies_bouldin_score(pred,preds)
    ch = calinski_harabasz_score(pred, preds)

    df['Cluster'] = kmeans.labels_
    json_records = df[:10].reset_index().to_json(orient ='records')
    arr = []
    arr = json.loads(json_records)
    column_names = list(df.columns.values)
    column_names.insert(0,df.index.name)
    context = {'d': arr , 'c': column_names, 'silhouette': silhouette, 'db': db, 'ch': ch }

    if request.method == 'GET':
        return render(request, 'authapp/autoencoder.html',context)
    if request.method == 'POST':
        result_csv = df.to_csv('./datasets/result_'+url[10:], encoding='utf-8')
        dataset = Dataset()
        dataset.dataset_name = url[10:]
        dataset.entreprise_name = 'ent1'
        dataset.dataset_path = 'datasets/result_'+url[10:]
        dataset.save()
        return render(request, 'authapp/autoencoder.html')




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
    context = {}
    if request.method == 'GET':
        dataset_list = Dataset.objects.all()
        context = {'list': dataset_list}
        return render(request, 'authapp/dashboardEmp.html',context)
    if request.method == 'POST':
        form = request.POST
        selected = request.POST['datasets']

        df = pd.read_csv(selected,index_col=0)
        index_name = df.index.name
        emp_id = request.user.empl.id
        row = df.loc[emp_id]
        cluster = row["Cluster"]

        df = df[df["Cluster"] == cluster]
        df = df[:10]
        df.columns = df.columns.str.replace(' ', '')
        column_names = list(df.columns.values)
        column_names.insert(0,df.index.name)
        json_records = df.reset_index().to_json(orient ='records')
        arr = []
        arr = json.loads(json_records)
        context = {'d': arr , 'c': column_names}
        return render(request, 'authapp/dashboardEmp.html',context)

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


@login_required(login_url='login')
@admin_only
def clusterEnsemble(request):
    url=request.GET.get('uploaded_file_url')
    df = pd.read_csv(url[1:],index_col=0)
    df.columns = df.columns.str.replace(' ', '')   
    scaler= MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    #PCA
    pca=PCA()
    pca.fit(df_scaled)
    pca_variance = pca.explained_variance_ratio_
    nb_features = len(df.columns)
    nbpc = 0
    varcumsum  = 0.0
    while varcumsum <= 0.8 :
        varcumsum = varcumsum + pca_variance[nbpc]
        nbpc = nbpc + 1
    
    pca=PCA(n_components=nbpc)
    pca.fit(df_scaled)
    scores_pca=pca.transform(df_scaled)
    km_scores= []
    for i in range(2,11):
        km = KMeans(n_clusters=i, random_state=0).fit(scores_pca)
        preds = km.predict(scores_pca)
        km_scores.append(km.inertia_)
    def optimal_number_of_clusters(wcss):
        x1, y1 = 2, wcss[0]
        x2, y2 = 10, wcss[len(wcss)-1]
        distances = []
        for i in range(len(wcss)):
            x0 = i+2
            y0 = wcss[i]
            numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(numerator/denominator)
        return distances.index(max(distances)) + 2
    nb_clusters = optimal_number_of_clusters(km_scores)
    kmeans_pca=KMeans(n_clusters=nb_clusters)
    kmeans_pca.fit(scores_pca)


    #AUTOENCODER
    train, test = train_test_split(df_scaled, test_size=0.2)
    nb_features = len(df.columns)
    input_df = Input( shape = (nb_features, ))
    x = Dense(round(nb_features * 0.8), activation = 'relu', kernel_initializer='glorot_uniform')(input_df)
    x = Dense(round(nb_features * 0.6), activation = 'relu', kernel_initializer='glorot_uniform')(input_df)
    x = Dense(round(nb_features * 0.4), activation = 'relu', kernel_initializer='glorot_uniform')(x)
    encoded = Dense(round(nb_features * 0.3), activation = 'relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(round(nb_features * 0.4), activation = 'relu', kernel_initializer='glorot_uniform')(encoded)
    x = Dense(round(nb_features * 0.6), activation = 'relu', kernel_initializer='glorot_uniform')(input_df)
    x = Dense(round(nb_features * 0.8), activation = 'relu', kernel_initializer='glorot_uniform')(x)
    decoded = Dense(nb_features, kernel_initializer='glorot_uniform')(x)
    autoencoder = Model(input_df, decoded)
    encoder = Model(input_df, encoded)
    autoencoder.save('autoencoder.h5')
    autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')
    history = autoencoder.fit(train, train, batch_size= 300, epochs = 35, verbose = 1, validation_data=(test,test))
    pred = encoder.predict(df_scaled)
    score = []
    range_values = range(2, 11)
    for i in range_values:
        kmeans = KMeans(n_clusters = i)
        kmeans.fit(pred)
        score.append(kmeans.inertia_)
    def optimal_number_of_clusters(wcss):
        x1, y1 = 2, wcss[0]
        x2, y2 = 10, wcss[len(wcss)-1]
        distances = []
        for i in range(len(wcss)):
            x0 = i+2
            y0 = wcss[i]
            numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            distances.append(numerator/denominator)
        return distances.index(max(distances)) + 2
    nb_clusters = optimal_number_of_clusters(score)
    kmeans = KMeans(nb_clusters)
    kmeans.fit(pred)

    labels_autoencoder=kmeans.labels_
    labels_pca_kmeans=kmeans_pca.labels_
    labels = np.array([labels_autoencoder,labels_pca_kmeans])
    label_com = CE.cluster_ensembles(labels)

    df['Cluster'] = label_com



    json_records = df[:10].reset_index().to_json(orient ='records')
    arr = []
    arr = json.loads(json_records)
    column_names = list(df.columns.values)
    column_names.insert(0,df.index.name)
    context = {'d': arr , 'c': column_names }

    if request.method == 'GET':
        return render(request, 'authapp/clusterEnsemble.html',context)
    if request.method == 'POST':
        result_csv = df.to_csv('./datasets/result_'+url[10:], encoding='utf-8')
        dataset = Dataset()
        dataset.dataset_name = url[10:]
        dataset.entreprise_name = 'ent1'
        dataset.dataset_path = 'datasets/result_'+url[10:]
        dataset.save()
        return render(request, 'authapp/clusterEnsemble.html')