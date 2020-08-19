"""
WSGI config for tens_obj_detection project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.2/howto/deployment/wsgi/
"""

# import os
#
# from django.core.wsgi import get_wsgi_application
#
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tens_obj_detection.settings')
#
# application = get_wsgi_application()


import os
import sys

from django.core.wsgi import get_wsgi_application
path = '/home/vishal/tens_obj_detection'   #path to project rckendoot
sys.path.append('/home/vishal/tens_obj_detection')
sys.path.append('/home/vishal/ml_env/bin/python3.6/site-packages')
os.environ["DJANGO_SETTINGS_MODULE"] = "tens_obj_detection.settings"

application = get_wsgi_application()