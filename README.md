# Image-Processing
Python-based image processing functions, leveraging OpenCV and NumPy for various transformations and effects.

# PROBLEM 1: LIGHT LEAK AND RAINBOW LIGHT LEAK

    This function takes 4 arguments, and writes the output to output1.jpg in the current directory. As a 
    pre-requisite, you must have the masks simple_mask.jpg and rainbow_mask.jpg in the current directory:

        - Input image: the name of the image file to which you would like to apply the light leak onto
            (this file should exist in the current directory)

        - Darkening Coefficient: a number between 0 and 1. The larger the number, the greater the darkening

        - Blending Coefficient: a number between 0 and 1. The larger the number, the greater the degree that
            the light light mask is blended into the image
        
        - Mode: This should be either 'simple' or 'rainbow'.

    *******************************************************************
    Example input to the command line to run Problem 1:
        Light Leak: python skww86.py problem1 face1.jpg 0.3 0.7 simple
        Rainbow Light Leak: python skww86.py problem1 face1.jpg 0.4 0.7 rainbow
    *******************************************************************

# PROBLEM 2: PENCIL / CHARCOAL EFFECT

    This function takes 3 arguments, and writes the output to output2.jpg in the current directory:

        - Input image: the name of the image file to which you would like to apply the effect onto
            (this file should exist in the current directory)

        - Blending Coefficient: a number between 0 and 1. The greater the number, the greater the degree that
            the noise texture is blended into the image

        - Mode: This should be either 'monochrome' or 'coloured_pencil'.

    *******************************************************************
    Example input to the command line to run Problem 2:
        Monochrome: python skww86.py problem2 face1.jpg 0.6 monochrome
        Coloured Pencil: python skww86.py problem2 face1.jpg 0.7 coloured_pencil
    *******************************************************************

# PROBLEM 3: SMOOTHING & BEAUTIFYING FILTER

    This function takes 3 arguments, and writes the output to output3.jpg in the current directory:

        - Input image: the name of the image file to which you would like to apply the effect onto
            (this file should exist in the current directory)

        - Size: a positive number set as the size (both width and height) of the kernel used for the
            Gaussian Blurring. Recommended value = 5

        - Sigma: a nonzero number set as the standard deviation (in both the x and y directions) for
            the Gaussian Blurring. Recommended value = 1

    *******************************************************************
    Example input to the command line to run Problem 3:
        python skww86.py problem3 face2.jpg 5 1
    *******************************************************************
    
# PROBLEM 4: FACE SWIRL

    This function takes 4 arguments, and writes 4 outputs to output4a.jpg, output4b.jpg, output4c.jpg,
    and output4d.jpg respectively in the current directory, where 4a is the image with the basic swirl
    effect applied to it, 4b is the image with the swirl effect applied to it where low pass filtering
    had been applied beforehand, 4c is the image resembling the source image whereby the inverse swirl
    has been applied to the swirled image, and 4d is the subtraction of the inverse swirled image from
    the original input image:

        - Input image: the name of the image file to which you would like to apply the effect onto
            (this file should exist in the current directory)

        - Strength: The strength of the swirl effect (swirl angle). Recommended value = 4

        - Radius: The radius of the swirl extent. Recommended value = 350

        - Interpolation: The interpolation method used. This should be either 'nearest_neighbour'
            or 'bilinear'. Note this is only relevant for tasks 1 and 2 (outputs 4a and 4b).

    *******************************************************************
    Example input to the command line to run Problem 3:
        python skww86.py problem4 face2.jpg 4 350 nearest_neighbour
    *******************************************************************
