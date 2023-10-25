from django.shortcuts import render,HttpResponse
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage


import os
import shutil
from .HyperSpectralAnalysis_VegationIndex.Test_20221109 import hyper

# Create your views here.
def home(request):
    response={}
    if request.method == 'POST' :
    #and request.FILES['myfile']:
        myfile_hdr = request.FILES['myfile_hdr']
        myfile_bsq = request.FILES['myfile_bsq']
        
        fs = FileSystemStorage()
        myfile0 = fs.save(myfile_hdr.name, myfile_hdr)
        myfile1 = fs.save(myfile_bsq.name, myfile_bsq)
        
        fileName = ''.join(list(myfile1)[:-4])

        response['fileName'] = myfile1
    
        uploaded_file_url = fs.url(myfile1)
        
        response['uploaded_file_url'] = uploaded_file_url

        # move file to target folder
        os.makedirs('media/file/', exist_ok=True)
        shutil.move('media/' + myfile0, 'media/file/' + myfile0)
        shutil.move('media/' + myfile1, 'media/file/' + myfile1)
        
        hyper('media/file/' + myfile0,'media/file/' + myfile1)
        pre2,ndvi = hyper('media/file/' + myfile0,'media/file/' + myfile1)
        
  
        response['pre2'] = pre2
        response['ndvi'] = ndvi
        response['image'] = 'welcome\\HyperSpectralAnalysis_VegationIndex\\NDVI.png'
        
    return render(request, 'home.html', response)
    
