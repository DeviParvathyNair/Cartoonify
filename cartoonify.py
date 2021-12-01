# Importing required libraries

import streamlit as st
from PIL import Image
import cv2 
import numpy as np
import tempfile
import os
import glob
from tqdm import tqdm
from os.path import isfile, join
from pathlib import Path

def main():
    # Defining sidebar options
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome', 'Upload file', 'Use webcam videofeed')
    )
    
    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'Upload file':
        upload()
    if selected_box == 'Use webcam videofeed':
        webcam()

def welcome():
    """
    Landing welcome page 
    """

    st.title("Cartoonify")

    st.markdown('An app that lets you experiment with different blurs and' 
    + 'filters to achieve different styles of cartoonification effects. '
    + 'Built in python using OpenCV and Streamlit.')

    
    images = Path("/images/")
    example = images/"example.jpg"
    #example = cv2.imread("./images/example.jpg")
    st.image(os.path.join("images/", "example.jpg"))

    st.markdown("""## Features

- Live previews
- Cross platform
- Picture/Video input support
- Live webcam input support


## Filter/Effects available

- Median filter
- Gaussian filter
- Bilateral filter
- Laplacian
- Adaptive Thresholding
- Bilateral filter
- Detail enhancing filters
- Erosion
- Colour Quantisation)
""")

    st.markdown("""
    ## Authors

    - Ann Maria John 
    - Athul Menon 
    - Devi Parvathy Nair
    """)

def upload():
    """
    Upload page
    """

    st.title('Cartoonify')
    
    st.subheader('An app that lets you experiment with different blurs and' 
    + 'filters to achieve different styles of cartoonification effects. '
    + 'Built in python using OpenCV and Streamlit.')
    

    # File uploader
    uploaded_file = st.file_uploader("Choose an image (or video)", type=["jpg","png","jpeg", "mp4", "mkv"])
    
    # Checking if a file is uploaded
    if uploaded_file is not None:
        if uploaded_file.name[-3:] == "mp4" or uploaded_file.name[-3:] == "mkv":
            
            # If a video is uploaded
            
            #print(uploaded_file.name[-3:])

            # Displaying the original video
            st.text("Original video")
            bytes_data = uploaded_file.read()
            st.video(bytes_data)

            # Reading it as a cv2 Video capture object
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(bytes_data)
            vf = cv2.VideoCapture(tfile.name)

            # Calling the cartoonify_video function
            cartoonify_video(vf)

        else:
            # If an image is uploaded

            # Opening our image
            image = Image.open(uploaded_file)
            image = np.array(image)
            
            #print(uploaded_file.name[-3:])

            # Removing alpha channel is the image is a png
            if uploaded_file.name[-3:] == "png":
                image = image[:,:,:-1]

             # Displaying the original picture
            st.text("Original image")
            st.image(image, use_column_width=True)
            #print(image.shape)
            cartoonify_withoptions(image)

def webcam():
    """
    If webcam input is chosen
    """

    # Decalring an image frame to display webcam input.
    FRAME_WINDOW = st.image([])
    videoCaptureObject = cv2.VideoCapture(0)

    # Declaring text input box
    path = st.text_input('Press q to capture an image')
    
    while videoCaptureObject.isOpened():
        _, frame = videoCaptureObject.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        # Continuosly collects frames till user gives input

        if path == 'q':
            image = frame
            break

    # ret, img = videoCaptureObject.read()
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # image = np.array(img_rgb)

    # Displaying image
    st.text("Original image")
    st.image(image, use_column_width=True)

    # Sends captured image to cartoonify
    webcam_cartoon = cartoonify_withoptions(image)

    videoCaptureObject.release()

    # Saving the cartoonified image
    if st.button("Save the cartoonified picture"):
        cv2.imwrite(os.path.join("images/", "webcam_output.jpg"), webcam_cartoon)


def cartoonify_video(vf):
    """"
    Function to split videos frame by frame, apply desired cartoonification
    effect and combining the images to recreate the video
    """

    # Choosing the required settings

    st.text("Choose required settings")
    
    blur = st.selectbox(
    'Enter the type of blur to be used',
    ('Median blur','Gaussian blur', 'Bilateral filter')
    )

    edge = st.selectbox(
        'Enter the type of edge detection to be used',
        ('Adaptive Thresholding', 'Laplacian')
    )

    erosion = st.selectbox(
        'Specify whether erosion is to be applied',
        ('No', 'Yes')
    )

    quant = st.selectbox(
        'Specify whether color quantisation is to be used for cartoonification',
        ('No', 'Yes')
    )

    # Default values in case user does not select
    k = 7
    filter = "Bilateral Filter"
    if quant == 'Yes':
        k = st.slider('Enter the number of clusters', min_value=5, max_value=10)
    if quant == 'No':
        filter = st.selectbox(
            'Enter the type of filter to be applied to the processed image',
            ('Bilateral Filter', 'Detail Enhancing Filter')
        )

    # Creating a data folder and removing its contents
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
        else:
            files = glob.glob('data/*')
            for f in files:
                os.remove(f)
    except OSError:
        print ('Error: Creating directory of data')

    # Finding the framerate of the video
    rec_fr = vf.get(cv2.CAP_PROP_FPS)
    print("Frame rate of video", vf.get(cv2.CAP_PROP_FPS))

    # For ideal framerate, but takes more time.
    frameRate = 1/rec_fr

    # For faster rendering
    #frameRate = 1
    
    sec = 0
    count = 0

    # Getting each frame from the video
    while(True):
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)

        vf.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        success, frame = vf.read()

        if success:
            frame_cartoon = cartoonify(frame, blur, edge, erosion, quant, k, filter)
            name = './data/frame' + str(count) + '.jpg'
            # Writing the images to /data
            cv2.imwrite(name, frame_cartoon)

            print(f"Wrote image {count}")
            
        else:
            break
        #st.image(frame_cartoon, use_column_width=True)

    #st.image(frame_cartoon, use_column_width=True)

    pathIn= './data/'
    pathOut = 'cartoon.mp4'
    size = 0
    
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # Sorting the file names properly
    files.sort(key = lambda x: x[5:-4])
    files.sort()

    for i in range(len(files)):
        filename=pathIn + files[i]
        #print(filename)
        
        # Reading each file
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
    
        # Inserting the frames into an image array
        frame_array.append(img)
        print(f"Appended image {i}")

    # Writing video
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), 1/frameRate, size)
    print("Video written")

    for i in range(len(frame_array)):
        # Writing to a image array
        out.write(frame_array[i])
    out.release()
 
    if os.path.exists('cartoon_output.mp4'):
        os.remove('cartoon_output.mp4')

    # Using ffmpeg to convert the video condec to be suitable for HTML 5
    os.system('ffmpeg -i {} -vcodec libx264 {}'.format(pathOut, 'cartoon_output.mp4'))

    # Displaying Cartoonified video
    st.text("Cartoonified video")
    video_file = open('cartoon_output.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

def cartoonify_withoptions(image):
    """
    Function to edit options of cartoonification effect in real time
    """
    st.header("Cartoonification")
        
    #image = cv2.imread('house.jpg')
    image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    st.text("Enter the type of blur to be used\n1. Median blur \n2. Gaussian blur\n3. Bilateral filter")
    
    blur = st.selectbox(
    'Enter the type of blur to be used',
    ('Median blur','Gaussian blur', 'Bilateral filter')
    )

    if blur == 'Median blur':
        #Applying median blur with kernel size of 5*5
        image_blur=cv2.medianBlur(image_gray, 5)
    if blur == 'Gaussian blur':
        #Applying gaussian blur with kernel size of 7*7
        image_blur=cv2.GaussianBlur(image_gray,(7,7),0)
    if blur == 'Bilateral filter':
        #Applying gaussian blur with kernel size of 5*5
        image_blur=cv2.bilateralFilter(image_gray, 5, 80, 80)

    st.image(image_blur, use_column_width=True,clamp = True)

    edge = st.selectbox(
    'Enter the type of edge detection to be used',
    ('Adaptive Thresholding', 'Laplacian')
    )

    if edge == 'Laplacian':
        # Edge retrieval using laplacian with kernel size of 5*5
       edges = cv2.Laplacian(image_blur, cv2.CV_8U, ksize=5)
       ret, image_edges = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    if edge == 'Adaptive Thresholding':
        # Edge retrieval using adaptive thresholding
        image_edges = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 9, 9)

    st.text("Edges of the smoothened image")
    st.image(image_edges, use_column_width=True,clamp = True)

    erosion = st.selectbox(
    'Specify whether erosion is to be applied',
    ('No', 'Yes')
    )

    if erosion == 'Yes':
        kernel=np.ones((2,2), np.uint8)
        #Performing erosion using kernel of size 2*2
        image_edges=cv2.erode(image_edges,kernel, iterations=1)

        st.text("Edges after Erosion")
        st.image(image_edges, use_column_width=True,clamp = True)

    if erosion == 'No':
        st.text("Edges without Erosion")
        st.image(image_edges, use_column_width=True,clamp = True)
    
    quant = st.selectbox(
    'Specify whether color quantisation is to be used for cartoonification',
    ('No', 'Yes')
    )

    if quant == 'Yes':

        #Defining the function for color quantisation
        def colorQuantization(image, k):
            #Defining input data for clustering by reshaping the input image to a 2D array of pixels
            #As cv2.kmeans() method takes in a 2D float array as input
            image_data = np.float32(image).reshape((-1, 3))
            #Artificially inducing stoping condition 
            condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            # Applying cv2.kmeans function to obtain the center points of the k clusters and label for each pixel
            _, label, center = cv2.kmeans(image_data, k, None, condition, 10, cv2.KMEANS_RANDOM_CENTERS)
            #Convertion of center points to 8 bit values
            center = np.uint8(center)
            #Label array is flattened and the pixels are convreted to the color of centroids
            result_image = center[label.flatten()]
            #Reshaping the image to the original dimension
            result_image = result_image.reshape(image.shape)
            return result_image
        
        k = st.slider('Enter the number of clusters',min_value = 5,max_value = 10)
        
        image_color_quantised=colorQuantization(image, k)
        st.text("Color Quantised Image")
        st.image(image_color_quantised, use_column_width=True,clamp = True)
        
        colorImage = cv2.bilateralFilter(image_color_quantised, d=7, sigmaColor=150,sigmaSpace=150)
        st.text("Color Quantised Image + Median Blur")
        st.image(colorImage, use_column_width=True,clamp = True)
        
    if quant == 'No':
        filter = st.selectbox(
            'Enter the type of filter to be applied to the processed image',
            ('Bilateral Filter','Detail Enhancing Filter')
        )

        if filter == 'Bilateral Filter':
            #Applies a bilateral image to a filter to sharpen the edges and smoothen the texture
            colorImage=cv2.bilateralFilter(image, 9, 300, 300)

        if filter == 'Detail Enhancing Filter':
            #Enhances details of the original image
            colorImage=cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)

        st.text("Image filter")
        st.image(colorImage, use_column_width=True,clamp = True)
    
    st.text("Cartoon Image")
    image_cartoon = cv2.bitwise_and(colorImage, colorImage, mask=image_edges)
    st.image(image_cartoon, use_column_width=True,clamp = True)
    
def cartoonify(image, blur, edge, erosion, quant, k, filter):
    """
    Funcation to cartoonify with preset options
    """
    #print(image.shape)
    image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if(blur=="Median blur" or blur=="1"):
        #Applying median blur with kernel size of 5*5
        image_blur=cv2.medianBlur(image_gray, 5)
    elif(blur=="Gaussian blur" or blur=="2"):
        #Applying gaussian blur with kernel size of 7*7
        image_blur=cv2.GaussianBlur(image_gray,(7,7),0)
    elif(blur=="Bilateral Filter" or blur=="3"):
        #Applying gaussian blur with kernel size of 5*5
        image_blur=cv2.bilateralFilter(image_gray, 5, 80, 80)

    if(edge=="Laplacian" or edge=="1"):
        edges = cv2.Laplacian(image_blur, cv2.CV_8U, ksize=5)
        ret, image_edges = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    elif(edge=="Adaptive Thresholding" or edge=="2"):
        image_edges = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    if(erosion=="Yes"):
        kernel=np.ones((2,2), np.uint8)
        #Performing erosion using kernel of size 2*2
        image_edges=cv2.erode(image_edges,kernel, iterations=1)

    if(quant=="Yes"):
        #Defining the function for color quantisation
        def colorQuantization(image, k):
            #Defining input data for clustering by reshaping the input image to a 2D array of pixels
            #As cv2.kmeans() method takes in a 2D float array as input
            image_data = np.float32(image).reshape((-1, 3))
            #Artificially inducing stoping condition 
            condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            # Applying cv2.kmeans function to obtain the center points of the k clusters and label for each pixel
            _, label, center = cv2.kmeans(image_data, k, None, condition, 10, cv2.KMEANS_RANDOM_CENTERS)
            #Convertion of center points to 8 bit values
            center = np.uint8(center)
            #Label array is flattened and the pixels are convreted to the color of centroids
            result_image = center[label.flatten()]
            #Reshaping the image to the original dimension
            result_image = result_image.reshape(image.shape)
            return result_image
        
        image_color_quantised=colorQuantization(image, k)
        colorImage = cv2.bilateralFilter(image_color_quantised, d=7, sigmaColor=150,sigmaSpace=150)

    elif(quant == "No"):
        if filter == 'Bilateral Filter':
            #Applies a bilateral image to a filter to sharpen the edges and smoothen the texture
            colorImage=cv2.bilateralFilter(image, 9, 300, 300)
        
        if filter == 'Detail Enhancing Filter':
            #Enhances details of the original image
            colorImage=cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)

    # Returning final cartoonified image
    image_cartoon = cv2.bitwise_and(colorImage, colorImage, mask=image_edges)
    return image_cartoon

if __name__ == "__main__":
    main()