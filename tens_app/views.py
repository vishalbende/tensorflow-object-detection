from django.shortcuts import render, redirect
from django.views.generic.edit import View
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from datetime import datetime
from django.contrib import messages

from ml_repo.models.research.obj_det import detect_objects
from utility.utils import image_resize_300x300


class ObjectDetectionView(View):

    def detect(request, *args, **kwargs):
        try:

            if request.method == 'POST':

                file = request.FILES['image']

                if not file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    messages.error(request, 'Invalid Image Format !!')
                    return render(request, 'upload_image.html', {})

                # file store
                fs = FileSystemStorage()
                fs.save(file.name, file)

                image_path = settings.BASE_DIR + '/media/' + str(file.name)

                # resize image
                resized_image = image_resize_300x300(image_path)

                # detect object in image and get detected object image
                image_obj = detect_objects(resized_image)

                # store image
                output_image_name = file.name + str(datetime.now()) + '.jpg'
                output_image_path = settings.BASE_DIR + '/media/' + output_image_name
                image_obj.save(output_image_path, "JPEG", quality=80, optimize=True, progressive=True)

                return render(request, 'upload_image.html', {"file": file.name, "output_file": output_image_name})

            return render(request, 'upload_image.html', {})

        except Exception as e:
            messages.error(request, 'Something went wrong !!')
            return render(request, 'upload_image.html')
