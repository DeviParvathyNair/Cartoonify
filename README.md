
# Cartoonify

An app that lets you experiment with different blurs and filters to achieve different styles of cartoonification effects. Built in python using OpenCV and Streamlit.


## Features

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
- Colour Quantisation

## Installation

Once the repository is cloned, to install all the python libraries required for cartoonify

```bash
  pip -r requirements.txt
```

ffmpeg is also required which can be installed using 
```bash
  sudo apt-get install ffmpeg
```
## Deployment

To deploy this project run

```bash
  streamlit run cartoonify.py
```

Once executed, the project provides a local and network URL that connects it to the port number [8501](http://localhost:8501/) which can be used to connect locally through a web browser.


## Demo
When inputting an image

![How to use the app](videos/demo.gif "Demo")

## Examples

When an image is given as input

![Example input and output](images/example.jpg "Colosseum example")

When a video is given as input

![Example input and output](videos/videoexample.gif "Video example")

### Median, Gaussian, Bilateral
  ![Median, Gaussian, Bilateral](images/Filters.jpg "Median, Gaussian, Bilateral")

### Laplacian, Adaptive
  ![Laplacian, Adaptive](images/EdgeDetection.jpg "Laplacian, Adaptive")

### With erosion, without erosion
  ![With erosion, without erosion](images/ErosionorNot.jpg "With erosion, without erosion")

### With color quantization, without color quantization
  ![With color quantization, without](images/ColorQuantisationorNot.jpeg "With color quantization, without color quantization")

### Bilateral, DetailEnhancement
  ![Bilateral, DetailEnhancement](images/Processing.jpg "Bilateral, DetailEnhancement")


# Authors

- Ann Maria John
- Athul Menon
- Devi Parvathy Nair