from django.shortcuts import render,HttpResponse
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage

import os
import shutil

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import shutil
import time
import concurrent.futures
import sys
import logging
import functools
from pathlib import Path
from subprocess import Popen, PIPE

from django.conf import settings

fmt = getattr(settings, 'LOG_FORMAT', None)
lvl = getattr(settings, 'LOG_LEVEL', logging.DEBUG)

logging.basicConfig(format=fmt, level=lvl)
logging.debug("Logging started on %s for %s" % (logging.root.name, logging.getLevelName(lvl)))

# Create your views here.
def home(request):
    response={}
    if request.method == 'POST' and request.FILES['file']:
        myfile = request.FILES['file']
        
        fs = FileSystemStorage()
        myfile = fs.save(myfile.name, myfile)

        fileName = ''.join(list(myfile)[:-4])

        response['fileName'] = myfile
        logging.info(f"File: {myfile}")
    
        uploaded_file_url = fs.url(myfile)
        response['uploaded_file_url'] = uploaded_file_url

        # move file to target folder
        output_dir = "media/output/"
        os.makedirs('media/file/', exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        shutil.move('media/' + myfile, 'media/file/' + myfile)
        process = Popen(["python3", "welcome/grape.py", f"media/file/{myfile}", output_dir, "0.03"], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        
        files = sorted(os.listdir(os.path.join(output_dir)))
        logging.info(f"Latest file: {files[-1]}")
        response['image'] = f"{output_dir}{files[-1]}"

    return render(request, 'home.html', response)
