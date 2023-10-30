from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse

from django.shortcuts import render
#from detection import main

import cv2
import os
import shutil
from cabbages.ai_model import get_SAHI_model

import logging
from django.conf import settings

fmt = getattr(settings, 'LOG_FORMAT', None)
lvl = getattr(settings, 'LOG_LEVEL', logging.DEBUG)

logging.basicConfig(format=fmt, level=lvl)
logging.debug("Logging started on %s for %s" % (logging.root.name, logging.getLevelName(lvl)))

sahi = get_SAHI_model()

# Create your views here.
def home(request):
    response={}
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        
        fs = FileSystemStorage()
        myfile = fs.save(myfile.name, myfile)

        fileName = ''.join(list(myfile)[:-4])

        response['fileName'] = myfile
    
        uploaded_file_url = fs.url(myfile)
        response['uploaded_file_url'] = uploaded_file_url

        # move file to target folder
        os.makedirs('media/file/', exist_ok=True)
        shutil.move('media/' + myfile, 'media/file/' + myfile)
        counts, response['image'] = sahi.inference('media/file/'+myfile, slice=True)
        
        response['Chinese_Cabbages'] = counts['Chinese cabbage']
        response['Broccoli'] = counts['broccoli']
        response['Cabbages'] = counts['cabbage']
        response['Cauliflower'] = counts['cauliflower']


        # image = cv2.imread('media/file/'+myfile)

        # save image
        # output_image = cv2.resize(draw_image, (0,0), fx=0.2, fy=0.2)
        # if cv2.imwrite('media/file/' + fileName + '_outputImage.jpg', output_image):
        #     # move file to target folder
        #     shutil.move('media/file/' + fileName + '_outputImage.jpg', 'media/output/' + fileName + '_outputImage.jpg')
        #
        #     # # remove files in folder
        #     shutil.rmtree('media/file')
        #     os.makedirs('media/file')
        #
        #     # must add a '/' ahead
        #     response['image'] = 'media/output/' + fileName + '_outputImage.jpg'

    return render(request, 'home.html', response)
