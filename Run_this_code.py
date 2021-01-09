import numpy as np
import os
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt
import unittest
from tensorflow.python.framework import ops

LR = 1e-3  # learning rate
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')


# name of the CNN model


class DogCat:

    def __init__(self, model_name, test_data):
        """
        This is the constructor
        it loads trained model, also loads images which need to be tested
        """
        self.img_size = 50

        #  following code are the neural network:
        #  it is a 5 layered convolutional neural network
        #  with a fully connected layer, and then the output layer

        convnet = input_data(shape=[None, self.img_size, self.img_size, 1], name='input')
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 128, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)
        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                             loss='categorical_crossentropy', name='targets')

        self._model = tflearn.DNN(convnet, tensorboard_dir='../Final project/log')

        if os.path.exists('{}.meta'.format(model_name)):
            self._model.load(model_name)  # load the trained model weights

        # load the testing image set
        self.test_data = np.load(test_data, allow_pickle=True)

    def show_example_figure(self):
        """
        This method inspects our network against unlabeled data visually
        We tested unlabeled first 12 images and plot the results
        """

        fig = plt.figure()

        for num, data in enumerate(self.test_data[:12]):
            # cat: [1,0]
            # dog: [0,1]
            # img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            model_out = self._predict(img_data)

            if np.argmax(model_out) == 1:
                str_label = 'Dog'
            else:
                str_label = 'Cat'

            y.imshow(orig, cmap='gray')
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)
        plt.show()

    def classify_images(self, num_images):
        """
        This method takes a number of images we want to classify
        Then, using our trained model to classify these images,
        it saves the result in a csv file,
        and return the number of images of dogs and cats
        """
        # create a csv file
        with open('classify_result_file.csv', 'w') as f:
            f.write('id,label\n')

        with open('classify_result_file.csv', 'a') as f:
            num_dog = 0
            num_cat = 0
            for data in self.test_data[0:num_images]:
                img_num = data[1]
                img_data = data[0]
                model_out = self._predict(img_data)
                f.write('{},{}\n'.format(img_num, model_out[1]))
                if np.argmax(model_out) == 1:
                    num_dog += 1
                else:
                    num_cat += 1
        return num_dog, num_cat

    def _predict(self, image):
        """
        This method takes an image as an argument
        then, reshape the image,
        using trained-model to return the predicted result
        """
        try:
            data = image.reshape(-1, self.img_size, self.img_size, 1)
            return self._model.predict(data)[0]

        except:
            return [0, 0]

    def __repr__(self):
        """
        this method returns a string describe what this class do
        """
        return 'This program classify dog and cat images!'


class UnitTestDogCat(unittest.TestCase):

    def unit_test(self):
        """
        This is a unit_test
        """
        test_data_fn = 'test_data.npy'
        dogcat = DogCat(MODEL_NAME, test_data_fn)
        # the unit test idea here is that if the first 12 images are 8 dogs 4 cats,
        # then it passes the test, otherwise, it will print the error message
        self.assertEqual(dogcat.classify_images(12), (8, 4),
                         'wrong number of dogs and cats')


def input_num_image():
    """
    This function will ask how many images user want to test
    Check requirements:
    --- iteration type (for, while),
    --- conditional (if)
    --- try blocks
    --- user-defined functions
    """
    print("We have 12500 unlabeled pictures, how many of them do you want to classify?\n"
          "Give me a number, after test, we will return how many dogs and cats are there")
    conditions = False
    while not conditions:
        n = input("Please enter an integer from 1 to 12500: ")
        try:
            num = int(n)
            if 0 < num <= 12500:
                conditions = True   # legal input
            else:
                print("Sorry, out of the range, please try again")
            continue
        except:
            print("This is not an integer, Please try again.")
    return num


# __main__ test:
if __name__ == '__main__':

    test_data_fn = 'test_data.npy'
    # create a object, pass the model and test date to this class
    dog_cat = DogCat(MODEL_NAME, test_data_fn)
    print(dog_cat)  # it will print repr()
    print("For example, we tested these 12 pictures in the testing set and plotted the results.\n")
    dog_cat.show_example_figure()  # plot the example testing figure with labels

    # call the function ask user to input how many image they want to test
    num_image = input_num_image()
    num_dog_cat = dog_cat.classify_images(num_image)
    print('We detected {} dogs and {} cats.'.format(num_dog_cat[0], num_dog_cat[1]))
    ops.reset_default_graph()
    unittest_num_dog_cat = UnitTestDogCat()
    unittest_num_dog_cat.unit_test()
