import sys

from .PDF2XML import convert_PDF2image
import matplotlib.pyplot as plt
import numpy as np
import cv2

from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from PIL import Image
from io import BytesIO


def array_to_pdf(array, filename):
    # Convertir le tableau NumPy en une image PIL
    image = Image.fromarray(array)

    # Redimensionner l'image pour s'adapter à la taille souhaitée
    new_width = len(array[0])  # Largeur souhaitée en pixels
    new_height = len(array)
    image = image.resize((new_width, new_height))

    # Créer un flux mémoire pour stocker temporairement l'image
    image_buffer = BytesIO()
    image.save(image_buffer, format='JPEG')

    # Créer un document PDF
    c = canvas.Canvas(filename, pagesize=(new_width, new_height))

    # Ajouter l'image depuis le flux mémoire au document PDF
    image_buffer.seek(0)  # Remettre le curseur au début du flux
    c.drawImage(ImageReader(image_buffer), 0, 0, width=new_width, height=new_height)

    # Enregistrer le document PDF
    c.save()

def anonymisation(file, out):
    dpi = 300
    images, page_number, _ = convert_PDF2image(file, DPI = dpi)
    image = np.array(images[0])
    #plt.imshow(image)
    
    # Convert the image in gray scale 
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian Blur
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

    ret,image_bin = cv2.threshold(img_blur,0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    # Dilate the image
    dilation = cv2.dilate(image_bin, rect_kernel, iterations = 1)
    # Find contour by applying rectangle
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im2 = image.copy()
    
    all_rect = []
    # For all the rectangles with a certain size mask them
    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt) 
        if x < 200 and y < 200:
            rect = cv2.rectangle(im2.astype('uint8'), (x, y), (x + w, y + h), (255, 0, 0), 2) 
            all_rect.append(rect)
            im2[y:y + h, x:x + w] = [255,255,255]
            

    

    array_to_pdf(im2,out)