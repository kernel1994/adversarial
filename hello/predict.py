import numpy as np
from keras.preprocessing import image
from keras.applications import xception


if __name__ == '__main__':
    # load pre-trained image recognition model
    model = xception.Xception()

    # load the image file and convert to a numpy array
    img = image.load_img('hacked_c.png', target_size=(299, 299))
    input_image = image.img_to_array(img)

    # scale the image so all pixel intensities are between [-1, 1] as the model expects
    input_image /= 255.
    input_image -= 0.5
    input_image *= 2.

    # add a 4th dimension for batch size (as Keras expects)
    input_image = np.expand_dims(input_image, axis=0)

    # run the image through the neural network
    predictions = model.predict(input_image)

    # convert the predictions into text and print them
    predicted_classes = xception.decode_predictions(predictions, top=1)
    imagenet_id, name, confidence = predicted_classes[0][0]
    print("This is a {} with {:.4}% confidence!".format(name, confidence * 100))
