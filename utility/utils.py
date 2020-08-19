from tensorflow.keras.preprocessing import image


# function to resize image by 300 by 300
def image_resize_300x300(image_path):
    # resize image
    img = image.load_img(image_path, target_size=(300, 300))
    image_array = image.img_to_array(img)

    # normalize and reshape
    image_array = image_array / 255
    image_array.reshape(1, 300, 300, 3)

    # array to image
    resized_image = image.array_to_img(image_array)
    return resized_image
