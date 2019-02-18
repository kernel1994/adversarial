# Machine Learning is Fun Part 8: How to Intentionally Trick Neural Networks by @ageitgey
# https://link.medium.com/W4KOswajkU
import sys
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.applications import xception
from keras import backend as K


if __name__ == '__main__':
    # Load pre-trained image recognition model
    model = xception.Xception()

    # Grab a reference to the first and last layer of the model
    model_input = model.layers[0].input  # Tensor, (batch_size, h, w, 3)
    model_output = model.layers[-1].output  # Tensor, (batch_size, 1000)

    # Choose an ImageNet object to fake
    # The list of classes is available here: https://gist.github.com/ageitgey/4e1342c10a71981d0b491e1b8227328b
    # Class #851 is "television"
    object_type_to_fake = 851

    # Load the image to hack
    img = image.load_img('c.png', target_size=(299, 299))
    original_image = image.img_to_array(img)

    # Scale the image so all pixel intensities are between [-1, 1] as the model expects
    original_image /= 255.
    original_image -= 0.5
    original_image *= 2.

    # Add a 4th dimension for batch size (as Keras expects)
    original_image = np.expand_dims(original_image, axis=0)

    # Pre-calculate the maximum change we will allow to the image
    # we will make sure our hacked image never goes past this so it does not look funny.
    # A larger number produces an image faster but risks more distortion
    max_change_above = original_image + 0.01
    max_change_below = original_image - 0.01

    # Create a copy of the input image to hack on
    hacked_image = np.copy(original_image)

    # How much to update the hacked image in each iteration
    learning_rate = 0.1

    # Define the cost function
    # Our 'cost' will be the likelihood out image is the target class according to the pre-trained model
    # 0 is batch size, object_type_to_fake is target class index of likelihood
    cost_function = model_output[0, object_type_to_fake]

    # We will as Keras to calculate the gradient based on the input image and the currently predicted class
    # In this case, referring to 'model_input_layer' will give us back image we hacking
    gradient_function = K.gradients(cost_function, model_input)[0]

    # Create a Keras function that we can call to calculate the current cost and gradient
    # Keras API:
    #   K.function(inputs, outputs, updates=None)
    #   Returns: Output values as Numpy arrays.
    #
    #   K.learning_phase()
    #   The learning phase flag is a bool tensor (0 = test, 1 = train) to be passed as input to any Keras function
    #   that uses a different behavior at train time and test time.
    grab_cost_and_gradients_from_model = K.function([model_input, K.learning_phase()],
                                                    [cost_function, gradient_function])

    cost = 0.0
    write = sys.stdout.write
    flush = sys.stdout.flush

    # In a loop, keep adjusting the hacked image slightly so that it tricks the model more and more
    # until ot gets to at least 80% confidence.
    while cost < 0.8:
        # Check how close the image is to our target class and grab the gradients
        # we can use to push it one more step in that direction.
        # Note: It's really important to pass in '0' for the Keras learning mode here!
        # Keras layers behave differently in prediction vs train modes!
        cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

        # Move the hacked image one step further towards fooling the model
        hacked_image += gradients * learning_rate

        # Ensure that the image dose not ever change too much either look funny or to become an invalid image
        hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
        hacked_image = np.clip(hacked_image, -1.0, 1.0)

        msg = "Model's predicted likelihood that the image is a television: {:.8}%".format(cost * 100)
        write(msg)
        flush()
        # Move cursor to the start of line
        write('\x08' * len(msg))

    # De-scale the image's pixels from [-1, 1] to the [0, 255] range
    img = hacked_image[0]
    img /= 2.
    img += 0.5
    img *= 255.

    # Save the hacked image
    im = Image.fromarray(img.astype(np.uint8))
    im.save('hacked_c.png')
