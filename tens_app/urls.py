from django.conf.urls import url
from tens_app.views import ObjectDetectionView

urlpatterns = [

    url('index/', ObjectDetectionView.detect, name='index'),


]
