import cv2
import sys
import numpy as np
import math
from scipy.interpolate import UnivariateSpline

# Helper function used in Problem 4
def swirl(img, new_image, strength, radius, interpolation):
    rows, cols, channels = new_image.shape
    # Centre co-ordinates of the swirl
    x_centre = round(rows/2)
    y_centre = round(cols/2)
    # Radius of the swirl extent
    radius = (np.log(2) * radius) / 5
    for x in range(rows):
        for y in range(cols):
            p = math.sqrt(((x-x_centre)**2) + ((y-y_centre)**2))
            angle = np.arctan2((y - y_centre), (x - x_centre)) + (np.exp(-p / radius) * strength)
            # Establish the mapped values from the original image
            u = round((p * math.cos(angle)) + x_centre)
            v = round((p * math.sin(angle)) + y_centre)
            if interpolation == 'nearest_neighbour':
                if u <= (rows-1) and v <= (cols-1):
                    # Take the value of the nearest point
                    new_image[x,y,0] = img[u,v,0]
                    new_image[x,y,1] = img[u,v,1]
                    new_image[x,y,2] = img[u,v,2]
            elif interpolation == 'bilinear':
                if u <= (rows-1) and v <= (cols-1):
                    # Establish the 4 nearest points along with their corresponding values
                    (x1, y1, s1), (_x1, y2, s2), (x2, _y1, s3), (_x2, _y2, s4) = sorted([(int(u),int(v),img[int(u),int(v),0]),(math.ceil(u),int(v),img[math.ceil(u),int(v),0]),(int(u),math.ceil(v),img[int(u),math.ceil(v),0]),(math.ceil(u),math.ceil(v),img[math.ceil(u),math.ceil(v),0])])
                    # Calculate the distances and values to these points and undertake the interpolation
                    new_image[x,y,0] = ((s1 * (x2 - u) * (y2 - v)) + (s3 * (u - x1) * (y2 - v)) + (s2 * (x2 - u) * (v - y1)) + (s4 * (u - x1) * (v - y1)))
                    (x1, y1, s1), (_x1, y2, s2), (x2, _y1, s3), (_x2, _y2, s4) = sorted([(int(u),int(v),img[int(u),int(v),1]),(math.ceil(u),int(v),img[math.ceil(u),int(v),1]),(int(u),math.ceil(v),img[int(u),math.ceil(v),1]),(math.ceil(u),math.ceil(v),img[math.ceil(u),math.ceil(v),1])])
                    new_image[x,y,1] = ((s1 * (x2 - u) * (y2 - v)) + (s3 * (u - x1) * (y2 - v)) + (s2 * (x2 - u) * (v - y1)) + (s4 * (u - x1) * (v - y1)))
                    (x1, y1, s1), (_x1, y2, s2), (x2, _y1, s3), (_x2, _y2, s4) = sorted([(int(u),int(v),img[int(u),int(v),2]),(math.ceil(u),int(v),img[math.ceil(u),int(v),2]),(int(u),math.ceil(v),img[int(u),math.ceil(v),2]),(math.ceil(u),math.ceil(v),img[math.ceil(u),math.ceil(v),2])])
                    new_image[x,y,2] = ((s1 * (x2 - u) * (y2 - v)) + (s3 * (u - x1) * (y2 - v)) + (s2 * (x2 - u) * (v - y1)) + (s4 * (u - x1) * (v - y1)))
            else:
                print("Error. Invalid interpolation method. You must input either 'nearest_neighbour' or 'bilinear'.")
                sys.exit()
    return new_image

def problem1(input_image, darkening_coeff, blending_coeff, mode):
    if darkening_coeff < 0 or darkening_coeff > 1:
        print("Error: Darkening coefficient must be between 0 and 1")
        return
    if blending_coeff < 0 or blending_coeff > 1:
        print("Error: Blending coefficient must be between 0 and 1")
        return
    # Read in input image
    img = cv2.imread(input_image)
    try:
        rows, cols, channels = img.shape
    except:
        print("Error: Invalid input image.")
        return
    # Read in mask
    if mode == 'simple':
        mask = cv2.imread('simple_mask.jpg')
    elif mode == 'rainbow':
        mask = cv2.imread('rainbow_mask.jpg')
    else:
        print("Error: Invalid mode. Mode must be either simple or rainbow.")
        return
    try:
        rows2, cols2, channels2 = mask.shape
    except:
        print("Error: Unable to read in mask")
        return
    # If necessary, resize the mask so that it is the same size as the input image
    if rows2 != rows or cols2 != cols:
        print("Resizing mask...")
        mask = cv2.resize(mask, (rows, cols))

    # Iterate through and transform each pixel. This is done by first applying the darkening,
    # and then blending in the given degree of the light leak mask. This is done for each of
    #  the three colour channels.
    for y in range(cols):
        for x in range(rows):
            # Darken
            img[y, x, 0] = max(0, (img[y, x, 0] - (darkening_coeff*255)))
            # Blend in mask
            img[y, x, 0] = min(255, (round(img[y, x, 0])+round(blending_coeff*(mask[y, x, 0]))))
            # Darken
            img[y, x, 1] = max(0, (img[y, x, 1] - (darkening_coeff*255)))
            # Blend in mask
            img[y, x, 1] = min(255, (round(img[y, x, 1])+round(blending_coeff*(mask[y, x, 1]))))
            # Darken
            img[y, x, 2] = max(0, (img[y, x, 2] - (darkening_coeff*255)))
            # Blend in mask
            img[y, x, 2] = min(255, (round(img[y, x, 2])+round(blending_coeff*(mask[y, x, 2]))))

    # Write out image to file
    cv2.imwrite("output1.jpg", img)
    return

def problem2(input_image, blending_coeff, mode):
    img = cv2.imread(input_image)
    # Step 1: turn image to grayscale
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        print("Error: Invalid input image.")
        return

    if mode == 'monochrome':        
        rows,cols = img.shape
        # Create an empty image with a grey background
        noise_texture = np.zeros((rows,cols), np.uint8)
        noise_texture[::] = 127

        # Apply Salt & Pepper noise to our empty image
        p = 0.4
        for x in range(rows):          
            for y in range(cols):
                s = np.random.uniform(0.0, 1.0)    
                if s < p/2:
                    noise_texture[x,y] = 255
                elif s < p:
                    noise_texture[x,y] = 0
        
        # Create the kernel
        kernel = np.zeros((10, 10))
        # Set the diagonal values in the kernel matrix to 1 (for diagonal motion blur) 
        np.fill_diagonal(kernel, 1)
        # Apply the kernel to the noise texture to blur the salt & pepper noise
        noise_texture = cv2.filter2D(noise_texture, -1, kernel/10) 
        
        # Blend the image with the noise texture
        img = (1-blending_coeff)*(img) + (blending_coeff)*(noise_texture)
        
        # Write out image to file
        cv2.imwrite('output2.jpg', img)
        return
    
    elif mode == 'coloured_pencil':
        # Convert grayscale image back to RGB
        img_col = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        rows, cols, channels = img_col.shape

        # Create 2 seperate noise textures to be applied to the green and red channels
        g_noise_texture = np.zeros((rows,cols), np.uint8)
        g_noise_texture[::] = 127
        r_noise_texture = np.zeros((rows,cols), np.uint8)
        r_noise_texture[::] = 157

        # Independently salt & pepper each noise texture using 2 seperate random variables
        p = 0.4
        for x in range(rows):
            for y in range(cols):
                g_rand = np.random.uniform(0.0, 1.0)
                r_rand = np.random.uniform(0.0, 1.0)
                if g_rand < p/4:
                    g_noise_texture[x,y] = 255
                elif g_rand < p:
                    g_noise_texture[x,y] = 0
                if r_rand < p/6:
                    r_noise_texture[x,y] = 0
                elif r_rand < p:
                    r_noise_texture[x,y] = 255

        # Create the kernel
        kernel = np.zeros((10, 10))
        # Set the diagonal values in the kernel matrix to 1 (for diagonal motion blur) 
        np.fill_diagonal(kernel, 1)
        # Apply the kernel to the noise textures to blur the salt & pepper noise
        g_noise_texture = cv2.filter2D(g_noise_texture, -1, kernel / 10)
        r_noise_texture = cv2.filter2D(r_noise_texture, -1, kernel / 10)
        
        # Blend the noise textures with the corresponding image colour channels 
        img_col[:,:,1] = ((1-blending_coeff)*(img_col[:,:,1]) + (blending_coeff)*(g_noise_texture))
        img_col[:,:,2] = ((1-blending_coeff)*(img_col[:,:,2]) + (blending_coeff)*(r_noise_texture))
        
        # Write out image to file
        cv2.imwrite('output2.jpg', img_col)
        return
    
def problem3(input_image, size, sigma):
    img = cv2.imread(input_image)
    try:
        rows, cols, channels = img.shape
    except:
        print("Error: Invalid input image.")
        return
    if size < 0:
        print('Error: Invalid size. Size must be positive.')
        return
    if sigma == 0:
        print('Error: Invalid sigma. Sigma cannot be 0')
        return
    
    # Apply Gaussian blur to smooth the image
    size = int(size/2)
    a, b = np.mgrid[-size:size+1, -size:size+1]
    t = np.exp(-((a**2 + b**2) / (2 * sigma**2))) * (1 / (2 * np.pi * sigma**2))
    img = cv2.filter2D(img, -1, t)

    # Split image into colour channels
    blue_channel = img[:,:,0]
    green_channel = img[:,:,1]
    red_channel = img[:,:,2]
    
    # Create unique colour curves to be applied to each channel
    blue_curve = (UnivariateSpline([0, 32, 64, 96, 128, 160, 192, 224, 255], [0, 52, 74, 93, 123, 150, 187, 221, 255]))(range(256))
    green_curve = (UnivariateSpline([0, 32, 64, 96, 128, 160, 192, 224, 255], [0, 30, 64, 101, 138, 180, 202, 229, 255]))(range(256))
    red_curve = (UnivariateSpline([0, 32, 64, 96, 128, 160, 192, 224, 255], [0, 28, 64, 106, 148, 200, 212, 234, 255]))(range(256))
    
    # Apply each colour curve to the corresponding channel
    blue_channel = cv2.LUT(blue_channel, blue_curve).astype(np.uint8)
    green_channel = cv2.LUT(green_channel, green_curve).astype(np.uint8)
    red_channel = cv2.LUT(red_channel, red_curve).astype(np.uint8)
    
    # Save the colour channels back into the image
    img[:,:,0] = blue_channel
    img[:,:,1] = green_channel
    img[:,:,2] = red_channel
 
    # Write out image to file
    cv2.imwrite('output3.jpg',img)
    return

def problem4(input_image, strength, radius, interpolation):
    img = cv2.imread(input_image)
    try:
        rows, cols, channels = img.shape
    except:
        print("Error: Invalid input image.")
        return
    # *********************************************************
    # Task 1: Applying basic swirl transformation
    # *********************************************************
    new_image_1 = np.zeros((rows,cols,3), np.uint8)
    new_image_1 = swirl(img, new_image_1, strength, radius, interpolation)
    cv2.imwrite('output4a.jpg', new_image_1)

    # *********************************************************
    # Task 2: Pre-filtering the image before transformation
    # *********************************************************
    new_image_2 = np.zeros((rows,cols,3), np.uint8)
    blurred_img = img
    # Apply the gaussian blur
    a, b = np.mgrid[-2:3, -2:3]
    t = np.exp(-((a**2 + b**2) / 2)) * (1 / (2 * np.pi))
    blurred_img = cv2.filter2D(blurred_img, -1, t)
    new_image_2 = swirl(blurred_img, new_image_2, strength, radius, interpolation)
    cv2.imwrite('output4b.jpg', new_image_2)

    # *********************************************************
    # Task 3: Inverse transformation
    # *********************************************************
    new_image_3 = np.zeros((rows,cols,3), np.uint8)
    x_centre = round(rows/2)
    y_centre = round(cols/2)
    # Radius of the swirl extent
    radius = (np.log(2) * radius) / 5
    for u in range(rows):
        for v in range(cols):
            p = math.sqrt(((u-x_centre)**2) + ((v-y_centre)**2))
            if p == 0:
                continue
            # Inverse function
            m = math.tan(-math.acos((u-x_centre)/p) - strength*np.exp(-p/radius))
            n = math.tan(math.acos((u-x_centre)/p) - strength*np.exp(-p/radius))
            
            # Multiple different values satisfy the equations so get hold of them all first
            x1 = ((-1)*(math.sqrt(p**2/(1+(m**2)))) + x_centre)
            y1 = ((-1)*(math.sqrt(p**2/(1+(1/(m**2))))) + x_centre)

            x2 = ((1)*(math.sqrt(p**2/(1+(m**2)))) + x_centre)
            y2 = ((1)*(math.sqrt(p**2/(1+(1/(m**2))))) + x_centre)

            x3 = ((1)*(math.sqrt(p**2/(1+(m**2)))) + x_centre)
            y3 = ((-1)*(math.sqrt(p**2/(1+(1/(m**2))))) + x_centre)

            x4 = ((-1)*(math.sqrt(p**2/(1+(m**2)))) + x_centre)
            y4 = ((1)*(math.sqrt(p**2/(1+(1/(m**2))))) + x_centre)

            x5 = ((-1)*(math.sqrt(p**2/(1+(n**2)))) + x_centre)
            y5 = ((-1)*(math.sqrt(p**2/(1+(1/(n**2))))) + x_centre)

            x6 = ((1)*(math.sqrt(p**2/(1+(n**2)))) + x_centre)
            y6 = ((1)*(math.sqrt(p**2/(1+(1/(n**2))))) + x_centre)

            x7 = ((1)*(math.sqrt(p**2/(1+(n**2)))) + x_centre)
            y7 = ((-1)*(math.sqrt(p**2/(1+(1/(n**2))))) + x_centre)

            x8 = ((-1)*(math.sqrt(p**2/(1+(n**2)))) + x_centre)
            y8 = ((1)*(math.sqrt(p**2/(1+(1/(n**2))))) + x_centre)

            # Perform the original transformation function on all these values
            p1 = math.sqrt(((x1-x_centre)**2) + ((y1-y_centre)**2))
            angle1 = np.arctan2((y1 - y_centre), (x1 - x_centre)) + (np.exp(-p1 / radius) * strength)
            u1 = round((p1 * math.cos(angle1)) + x_centre)
            v1 = round((p1 * math.sin(angle1)) + y_centre)

            p2 = math.sqrt(((x2-x_centre)**2) + ((y2-y_centre)**2))
            angle2 = np.arctan2((y2 - y_centre), (x2 - x_centre)) + (np.exp(-p2 / radius) * strength)
            u2 = round((p2 * math.cos(angle2)) + x_centre)
            v2 = round((p2 * math.sin(angle2)) + y_centre)
            
            p3 = math.sqrt(((x3-x_centre)**2) + ((y3-y_centre)**2))
            angle3 = np.arctan2((y3 - y_centre), (x3 - x_centre)) + (np.exp(-p3 / radius) * strength)
            u3 = round((p3 * math.cos(angle3)) + x_centre)
            v3 = round((p3 * math.sin(angle3)) + y_centre)

            p4 = math.sqrt(((x4-x_centre)**2) + ((y4-y_centre)**2))
            angle4 = np.arctan2((y4 - y_centre), (x4 - x_centre)) + (np.exp(-p4 / radius) * strength)
            u4 = round((p4 * math.cos(angle4)) + x_centre)
            v4 = round((p4 * math.sin(angle4)) + y_centre)

            p5 = math.sqrt(((x5-x_centre)**2) + ((y5-y_centre)**2))
            angle5 = np.arctan2((y5 - y_centre), (x5 - x_centre)) + (np.exp(-p5 / radius) * strength)
            u5 = round((p5 * math.cos(angle5)) + x_centre)
            v5 = round((p5 * math.sin(angle5)) + y_centre)

            p6 = math.sqrt(((x6-x_centre)**2) + ((y6-y_centre)**2))
            angle6 = np.arctan2((y6 - y_centre), (x6 - x_centre)) + (np.exp(-p6 / radius) * strength)
            u6 = round((p6 * math.cos(angle6)) + x_centre)
            v6 = round((p6 * math.sin(angle6)) + y_centre)

            p7 = math.sqrt(((x7-x_centre)**2) + ((y7-y_centre)**2))
            angle7 = np.arctan2((y7 - y_centre), (x7 - x_centre)) + (np.exp(-p7 / radius) * strength)
            u7 = round((p7 * math.cos(angle7)) + x_centre)
            v7 = round((p7 * math.sin(angle7)) + y_centre)

            p8 = math.sqrt(((x8-x_centre)**2) + ((y8-y_centre)**2))
            angle8 = np.arctan2((y8 - y_centre), (x8 - x_centre)) + (np.exp(-p8 / radius) * strength)
            u8 = round((p8 * math.cos(angle8)) + x_centre)
            v8 = round((p8 * math.sin(angle8)) + y_centre)
            
            # Establish the correct values by checking equality
            if u == u1 and v == v1:
                x = x1
                y = y1
            elif u == u2 and v == v2:
                x = x2
                y = x2
            elif u == u3 and v == v3:
                x = x3
                y = y3
            elif u == u4 and v == v4:
                x = x4
                y = y4
            elif u == u5 and v == v5:
                x = x5
                y = x5
            elif u == u6 and v == v6:
                x = x6
                y = y6
            elif u == u7 and v == v7:
                x = x7
                y = y7
            elif u == u8 and v == v8:
                x = x8
                y = y8
            else:
                continue
            
            # Set the values in the new image
            if x <= (rows-1) and y <= (cols-1):
                new_image_3[u,v,0] = new_image_1[round(x),round(y),0]
                new_image_3[u,v,1] = new_image_1[round(x),round(y),1]
                new_image_3[u,v,2] = new_image_1[round(x),round(y),2]
    cv2.imwrite('output4c.jpg', new_image_3)

    # Subtract the 2 images
    new_image_4 = img - new_image_3
    cv2.imwrite('output4d.jpg', new_image_4)
    return

if __name__ == "__main__":
    if sys.argv[1] == 'problem1':
        problem1(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5])
    elif sys.argv[1] == 'problem2':
        problem2(sys.argv[2], float(sys.argv[3]), sys.argv[4])
    elif sys.argv[1] == 'problem3':
        problem3(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
    elif sys.argv[1] == 'problem4':
        problem4(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5])