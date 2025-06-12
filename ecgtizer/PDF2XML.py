import numpy as np
from pdf2image import convert_from_path, exceptions
from .extraction_functions import *
import cv2
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
#import pytesseract
import re
import pandas as pd

import io
import base64





def convert_PDF2image(path_input, DPI):
    
    """
    Convert the PDF file into array (images).
    
    We use the library pdf2image to transform the input file into an array
    
    Parameters
    ----------
    path_input : str, path of the pdf file to convert
    DPI :int, dots per inch (resolution of the image)
    
    Returns
    -------
    list : list of all the pages of the PDF in PIL format
    int  : number of pages
    bool : True: The conversion has worked / False :  The conversion has not worked
    """
    try:
        # Convert all the pages of the pdf into PIL
        pages = convert_from_path(path_input, poppler_path= '', dpi = DPI) 
    except exceptions.PDFPageCountError:
        print("Impossible conversion.\nThe input file is not a PDF.\n")
        return("_", "_", False)
    return(pages,len(pages), True)



def check_noise_type(image, DPI, DEBUG):
    
    """
    Check the noise level of the image. Check the type of the image.
    
    Parameters
    ----------
    image : np.array, image
    DPI   : int, dots per inch (resolution of the image)
    DEBUG : bool, show the image
    
    Returns
    -------
    str : Type of image
    bool : True: The image is noised / False : The image is not noised
    """
    
    # Check the color diversity
    liste = []
    for i in range(len(image)):
        #for j in range(len(image[i])):
            if image[i][int(len(image[i])/2)][1] not in liste and image[i][int(len(image[i])/2)][0] == 255 or image[i][int(len(image[i])/2)][2] not in liste and image[i][int(len(image[i])/2)][0] :
                liste.append(image[i][int(len(image[i])/2)][1])
                
    # Kardia format is in black and white
    if len(liste) == 1:
        return("Kardia", False)
    
    # Check the variance in the image 
    if np.var(image) > 2000 or np.var(image) < 600 :
        if np.var(image) > 3000:
            # above a variance of 3000 the image is considered noisy
            NOISE = True
        
        else:
            # below a variance we don't know if the image is noised or not
            NOISE = 0.5
    
    else:
        # below a variance of 600 the image is considered noisy  
        NOISE = False

    if len(image) > len(image[0]):
        # the Wellue format offers images that are taller than they are wide
        return("Wellue", NOISE)
    
    else:
        # Convert image in gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Binarize the image
        ret, thresh1 = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        # Define the rectangle original size
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.03*len(image)),int(0.03*len(image))))
        # Dilate the image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
        # Find contour by applying rectangle
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        im2 = image.copy()
        nbr = 0
        # Count the number of rectangles in the apple watch format there is 3 record rectangle
        for cnt in contours: 
            x, y, w, h = cv2.boundingRect(cnt) 
            if w - x > len(image)/3:
                rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 2) 
                nbr +=1
        # Plot the image with the different rectangle(s) find
        if DEBUG == True:
            try:
                plt.figure(figsize = (20,14))
                plt.imshow(rect)
                plt.show()
            except UnboundLocalError:
                pass
        # There are more than 3 record rectangle it is apple watch format
        if nbr >= 3:
            return('apple', False)
        # There is less than 3 record rectangle it is a classical format
        else:
            return('classic', NOISE)
        
        
def text_extraction(image,page, DPI, NOISE, TYPE,  DEBUG):
    
    """
    Extract the texte from the image and mask the task on the image
    For Kardia it mask the gride line
    
    Parameters
    ----------
    image : np.array, image
    DPI   : int, dots per inch (resolution of the image)
    NOISE : bool, if the image is noised or not
    TYPE  : str, format of the image
    DEBUG : bool, show the image
    
    Returns
    -------
    array : The image without the text
    DataFrame : The dataframe with the extracted text in it
    """
    dic = {
    "Patient_ID" :"Empty",
    "age":"Empty",
    "sex":"Empty",
    "Date" :"Empty",
    "Hour":"Empty",
    "Base_ID":"Empty",
    "D_birth":"Empty",
    "Vent. Freq.":"Empty",
    "Int. PR":"Empty",
    "QRS dur.":"Empty",
    "QT":"Empty",
    "QTc":"Empty",
    "Axe P":"Empty",
    "Axe R":"Empty",
    "Axe T":"Empty",
    "Mean RR":"Empty",
    "QTcB":"Empty",
    "QTcF":"Empty",
    "Other":"Empty",
}
    df = [] 
    
    if TYPE.lower() == 'kardia':
        # Isolate the record region
        work_image = np.array(image)[DPI:int(10*DPI),int(0.3*DPI):int(8*DPI)]
        # Convert the image in gray scale
        image_gray = cv2.cvtColor(work_image,cv2.COLOR_BGR2GRAY) 
        # Binarize the image thanks to the gray scale
        new_image = np.full(image_gray.shape, 0)
        for c in range(len(image_gray)):
            for l in range(len(image_gray[c])):
                if image_gray[c,l] == 0:
                        new_image[c,l] = 255
        


        # Compute the vertical variance
        var_line = np.var(new_image, axis= 1)
        # Compute the horizontal variance
        var_column = np.var(new_image, axis = 0)
        # Define a second image to work with
        working_image = np.copy(new_image)

        
        
        for i in range(len(new_image)):
            if var_line[i] < 1000:
                working_image[i,:] = 0
        for i in range(len(new_image[0])):
            if var_column[i] < 200:
                working_image[:,i] = 0
        
        if DEBUG == True:
            plt.figure(figsize = (20,14))
            plt.imshow(working_image)
            plt.show()
        return(working_image, df)
    
    # Table with the information patient in it
    
    
    # Convert image in gray scale
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian Blur
    image_blur = cv2.GaussianBlur(image_gray, (5,5), 0) 
    # If the image is noised we apply a deterministic threshold
    if NOISE == True or NOISE == 0.5: 
        # Binarize the image with the deterministic threshold
        ret, image_bin = cv2.threshold(image_gray, 40, 100, cv2.THRESH_BINARY_INV)
        # Compute the horizontal variance
        horizontal_variance = np.var(image_bin, axis = 1)
        # Detect the variance peaks
        peaks = signal.argrelextrema(horizontal_variance, np.greater, order = int(len(image)/10))[0] # Compute the pikes position
        # starting position on the x-axis
        x = 0
        # Ending position on the x-axis
        w = len(image[0])
        # Starting position on the y-axis
        y = 0
        # Ending position on the y-axis
        h = int(peaks[0] + (peaks[1]-peaks[0])/2)
        im2 = image.copy()
        # Define and apply a mask on the text region
        rect = cv2.rectangle(image_bin, (x, peaks[0]), (x + w, y + h), (255, 0, 0), 2)
        # The mask must have the same color as the rest of the image
        image[y:y + h, x:x + w] = np.mean(image[y:y + h, x:x + w])
        
    # If the image is not noised we apply a Otsu detection threshold    
    else:
        # Binarize the image with the Otsu threshold
        ret,image_bin = cv2.threshold(image_blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
        # Define the rectangle original size
        if TYPE == "apple":
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.03*len(image)),int(0.03*len(image))))
        else:
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.0075*len(image)),int(0.0075*len(image))))
        # Dilate the image
        dilation = cv2.dilate(image_bin, rect_kernel, iterations = 1)
        # Find contour by applying rectangle
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
        im2 = image.copy()
        
        # For all the rectangles with a certain size mask them
        for cnt in contours: 
            x, y, w, h = cv2.boundingRect(cnt) 
            if len(image) < len(image[0]): 
                if w-x < len(im2)/3:
                    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 2) 
                    image[y:y + h, x:x + w] = (255,255,255)
            else:
                if h-y < len(im2[0])/4:
                    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 2) 
                    image[y:y + h, x:x + w] = np.mean(image[y:y + h, x:x + w])
        
        # if page == 0:
        #     texte = pytesseract.image_to_string(image_bin)
        #     texte_split = re.split(r'\s+|\n', texte)
        #     it = 0
        #     liste_put = ["ms", "/"]
        #     for w in range(len(texte_split)):

        #         if it == 0:
        #             dic["Patient_ID"] = texte_split[w]
        #         elif it == 1:
        #             dic["Patient_ID"] = dic["Patient_ID"]+texte_split[w]

        #         elif it == 2:
        #             dic["Date"] = texte_split[w]
        #         elif it == 3:
        #             dic["Hour"] = texte_split[w]
        #         elif texte_split[w] == 'ID:':
        #             dic["Base_ID"] = texte_split[w+1]
        #             liste_put.append(texte_split[w+1])
        #         elif texte_split[w] == 'D-naiss:':
        #             dic["D_birth"] = texte_split[w+1]
        #             liste_put.append(texte_split[w+1])
        #         elif texte_split[w] == 'Fréq.Vent:':
        #             dic["Vent. Freq."] = texte_split[w+1] + "ms"
        #             liste_put.append(texte_split[w+1])
        #         elif texte_split[w] == 'PR:':
        #             dic["Int. PR"] = texte_split[w+1] + "ms"
        #             liste_put.append(texte_split[w+1])
        #         elif texte_split[w] == 'Dur.QRS:':
        #             dic["QRS dur."] = texte_split[w+1] + "ms"
        #             liste_put.append(texte_split[w+1])
        #         elif texte_split[w] == 'QT/QTc:':
        #             dic["QT"] = texte_split[w+1] + "ms"
        #             liste_put.append(texte_split[w+1])
        #             dic["QTc"] = texte_split[w+3] + "ms"
        #             liste_put.append(texte_split[w+3])
        #         elif texte_split[w] == 'P-R-T:':
        #             dic["Axe P"] = texte_split[w+1] + "ms"
        #             liste_put.append(texte_split[w+1])
        #             dic["Axe R"] = texte_split[w+2] + "ms"
        #             liste_put.append(texte_split[w+2])
        #             dic["Axe T"] = texte_split[w+3] + "ms"
        #             liste_put.append(texte_split[w+3])
        #         elif texte_split[w] == 'RR:':
        #             dic["Mean RR"] = texte_split[w+1] + "ms"
        #             liste_put.append(texte_split[w+1])

        #         elif texte_split[w][-4:] == 'ans,':
        #             dic["age"] = texte_split[w] 

        #         elif texte_split[w] == 'Fem.' or texte_split[w] == 'Male':
        #             dic["sex"] = texte_split[w] 

        #         elif texte_split[w] == 'QTcB:':
        #             dic["QTcB"] = texte_split[w+1] + "ms"
        #             liste_put.append(texte_split[w+1])
        #         elif texte_split[w] == 'QTcF:':
        #             dic["QTcF"] = texte_split[w+1] + "ms"
        #             liste_put.append(texte_split[w+1])

        #         elif texte_split[w] == 'QTcF:':
        #             dic["QTcF"] = texte_split[w+1] + "ms"
        #             liste_put.append(texte_split[w+1])



        #         else :
        #             if dic["Other"] == "Empty": 
        #                 dic["Other"] = texte_split[w]
        #             else:
        #                 if texte_split[w] not in liste_put:
        #                     dic["Other"] = dic["Other"] + " " + texte_split[w]


        #         it +=1


        #     df = pd.DataFrame(dic, index = [0]).T
    # Plot the image with the detected rectangles
    if DEBUG == True:
        try:
            plt.figure(figsize = (20,14))
            plt.imshow(rect)
            plt.show()
            plt.imshow(image)
            plt.show()
        except UnboundLocalError:
            plt.imshow(image)
            plt.show()
    return(image)       
    #return(image, dic)



def tracks_extraction(image, TYPE, DPI, FORMAT, NOISE = False, DEBUG = False):
    
    """
    Extract the tracks from the image
    
    Parameters
    ----------
    image : np.array, image
    TYPE  : str, format of the image
    DPI   : int, dots per inch (resolution of the image)
    FORMAT: str, multi or unilead for Kardia 
    NOISE : bool, if the image is noised or not
    DEBUG : bool, show the image
    
    Returns
    -------
    dictionary  : dictionary of the different extracted tracks with their position 
                  (key : position / Value: Track images)
    """
    # dictionary of all tracks
    dic_tracks = {}
    if TYPE.lower() == 'kardia':
        var_line = np.var(image, axis= 1)
        peaks,_ = find_peaks(var_line, height = 2*DPI, distance = DPI)
        start = 0
        it = 0
        for p in range(len(peaks)-1):
            end = (peaks[p]+peaks[p+1])/2
            dic_tracks[it] = image[start:int(end),:]
            start = int(end)
            it +=1
        dic_tracks[it] = image[start:,:]

        dic_tracks_temp = {}
        it =0
        if FORMAT == 'unilead':
            for i in dic_tracks:
                if i%2 == 0:
                    dic_tracks_temp[it] = dic_tracks[i]
                    it+=1
            if DEBUG == True:
                for im in dic_tracks_temp:
                    plt.imshow(dic_tracks_temp[im])
                    plt.show()
            return(dic_tracks_temp)
        
        else: 
            if DEBUG == True:
                for im in dic_tracks:
                    plt.imshow(dic_tracks[im])
                    plt.show()
            return(dic_tracks)
    
    # Plot the original image 
    if DEBUG == True:
        plt.figure(figsize = (20,14))
    
    # Convert the image in gray scale 
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian Blur
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    
    # If the image is noised we will use the Sauvola detection thresholding
    if NOISE != False: 
        # # Size of the local window for the Sauvola thresholding 
        # WINDOW_SIZE = 5 
        # # Apply Sauvola Thresholding
        # thresh_sauvola = threshold_sauvola(img_blur, window_size=WINDOW_SIZE)
        
        # # Binarize the image
        # image_bin = img_blur < thresh_sauvola 
        # image_bin2 = np.ones((len(image_bin),len(image_bin[0])))
        # for i in range(len(image_bin)):
        #     for j in range(len(image_bin[i])):
        #         if image_bin[i][j] == False:
        #             image_bin2[i][j] = 0
        #         else :
        #             image_bin2[i][j] = 255
        # image_bin = image_bin2.astype("uint8")
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # Binarize the image with the deterministic threshold
        ret, image_bin = cv2.threshold(image_gray, 40, 255, cv2.THRESH_BINARY_INV)
        
    # If the image is Wellue type we have determine the optimal threshold
    elif TYPE.lower() == "wellue":
        ret, image_bin = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)
        
    # If the image is not noised we will use the Otsu thresholding    
    else: 
        # Apply Otsu Thresholding
        ret,image_bin = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
    

    # Compute the horizontal variance on binarized image
    horizontal_variance     = np.var(image_bin, axis = 1) 
   
    
    # If images are taller than they are wide the distance between two peaks is smaller
    if len(image) > len(image[0]):
        # Compute the pikes position
        peaksh = signal.argrelextrema(horizontal_variance, np.greater, order = int(0.010*len(image)))[0] 
        
    # If images are wide than they are taller the distance between two peaks is bigger
    if len(image) < len(image[0]):
        # Compute the pikes position
        #peaks = signal.argrelextrema(horizontal_variance, np.greater, order = int(0.05*len(image)))[0] 
        peaksh, _ = find_peaks(horizontal_variance, height=len(image[0]), distance=int(len(image)/10))
        # if NOISE != False:
        #     peaksh, _ = find_peaks(horizontal_variance, height=(len(image)-(len(image[0])*15/100),len(image[0])), distance=int(len(image)/10))
        

    
    if DEBUG == True:
        plt.plot(horizontal_variance)
        for p in peaksh:
            plt.axvline(p, c = "r")
        plt.savefig("Horrizontal_variance.png")
        plt.show()
        plt.figure(figsize = (20,20))
        
    
    # Define a list with all the position to cut between tracks a we store the beggining of the image
    cut_pos = [0]
    # for all peaks we only keep the position between them
    for p in range(len(peaksh)-1):
        cut_pos.append(int((peaksh[p]+peaksh[p+1])/2))
    # We store the ending of the image
    cut_pos.append(len(image))
    
    # If we have 6 tracks we have extracted text information
    if len(cut_pos) == 6:
        del cut_pos[0]
    
    # We store all track image in the dictionary 
    it = 1
    for c in range(len(cut_pos)-1):
        if it == 1:
            dic_tracks[c] = image_bin[cut_pos[c]+int(0.05*len(image)):cut_pos[c+1]]
        elif it == len(cut_pos)-1:
            dic_tracks[c] = image_bin[cut_pos[c]:cut_pos[c+1]-int(0.09*len(image))]
        else:
            dic_tracks[c] = image_bin[cut_pos[c]:cut_pos[c+1]]
            
        it+=1
        
        if DEBUG == True:
            plt.axhline(cut_pos[c], c = 'g', alpha = 0.6)
            
    # Plot the position of the cut in the image 
    if DEBUG == True:
        plt.imshow(image)
        plt.axhline(cut_pos[-1], c = 'g', alpha = 0.6)
        for p in peaksh:
            plt.axhline(p, c = 'r', alpha = 0.6)
        
    # Compute the vertical variance   
    vertical_variance = np.var(image_bin, axis = 0) 

    # Define a list which will contain the pikes position
    peaksv = [] 
    for var in range(len(vertical_variance)):
        # If the variance is no null then there is a signal waveform a we must not cut here the signal
        if vertical_variance[var] > 200 : 
            # Pikes take the beggining position of the waveform
            peaksv.append(var)
    
    
            
    # For all the tracks we cut vertically the part which not contain waveform
    for track in dic_tracks.keys(): 
        dic_tracks[track] = dic_tracks[track][:,peaksv[0]:peaksv[-1]]
    
    # Plot the position of the cut in the image 
    if DEBUG == True:
        plt.axvline(peaksv[0])
        plt.axvline(peaksv[-1])
        plt.savefig("Image_of_tracks.png")
        plt.show()
        
    if DEBUG == True:
        plt.plot(vertical_variance)
        
        
        plt.axvline(peaksv[0], c = "r")
        plt.axvline(peaksv[-1], c = "r")
        plt.savefig("Vertical_variance.png")
        plt.show()
    
    
    return(dic_tracks, peaksh, peaksv[0])




def clean_tracks(dic_tracks, TYPE, NOISE, DEBUG):
    
    """
    Detect all groups of pixels and remove them
    
    Parameters
    ----------
    dic_tracks: dictionary, dictionary of track images
    TYPE  : str, format of the image
    NOISE : bool, if the image is noised or not
    DEBUG : bool, show the image
    
    Returns
    -------
    None
    """
    
    for d in dic_tracks:
        # Kardia files are already binarize
        if TYPE.lower() != 'kardia':
            # Convert the image in gray scale 
            img_gray = cv2.cvtColor(dic_tracks[d],cv2.COLOR_BGR2GRAY)
            # Apply a Gaussian Blur
            img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

            # If the image is Wellue type we have determine the optimal threshold
            if TYPE == 'Wellue':
                ret, image_bin = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)

            else:
                # If the image is noised we will use the Sauvola detection thresholding
                if NOISE == True: 
                    # Size of the local window for the Sauvola thresholding 
                    WINDOW_SIZE = 3 
                    # Apply Sauvola Thresholding
                    thresh_sauvola = threshold_sauvola(img_blur, window_size=WINDOW_SIZE)

                    # Binarize the image 
                    image_bin = img_blur < thresh_sauvola 
                    image_bin2 = np.ones((len(image_bin),len(image_bin[0])))
                    for i in range(len(image_bin)):
                        for j in range(len(image_bin[i])):
                            if image_bin[i][j] == False:
                                image_bin2[i][j] = 0
                            else :
                                image_bin2[i][j] = 255
                    image_bin = image_bin2

                # If the image is not noised we will use the Otsu thresholding   
                else: 
                    # Apply Otsu detection thresholding
                    ret,image_bin = cv2.threshold(img_blur,0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
            # Define the rectangle original size
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.045*len(dic_tracks[d])),int(0.045*len(dic_tracks[d]))))
            
        else:
            image_bin = dic_tracks[d].astype('uint8')
            # Define the rectangle original size
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
        

        # Dilate the image
        dilation = cv2.dilate(image_bin, rect_kernel, iterations = 1)
        # Find contour by applying rectangle
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        im2 = dic_tracks[d].copy()
        
        # For all the rectangles with a certain size mask them
        for cnt in contours: 
            x, y, w, h = cv2.boundingRect(cnt) 
            if w - x < 100 and h - y < 100:
                rect = cv2.rectangle(im2.astype('uint8'), (x, y), (x + w, y + h), (255, 0, 0), 2) 
                dic_tracks[d][y:y + h, x:x + w] = np.mean(dic_tracks[d][y:y + h, x:x + w])

        # Plot the image and the associated masks
        if DEBUG == True:
            plt.figure(figsize = (20,14))
            try:
                plt.imshow(rect)
            except UnboundLocalError:
                plt.imshow(im2)
            plt.show()
        
def sup_holes(signal, TYPE):
    
    """
    Fill the holes in the extracted signal
    
    Parameters
    ----------
    signal: array, contain the extracted signal
    TYPE: str, it can be :
            - "classic"
            - "heartcheck"
            - "duoek"
    
    Returns
    -------
    list: list of the extracted signal without hole
    """
    
    # if the signal is constant then we set the signal to 0
    if np.all(np.diff(signal) == 0):
        signal = np.zeros(len(signal)) # Else the signal is set to 0
        return(signal)
    
    end = -1
    # If the first value is a hole then we will search for the following point 
    # that is a point of signal and we will take its value
    if signal[0] == 0: 
        j = 1
        while signal[j] == 0:
            j+=1
        signal[0] = signal[j]
     
    # If the last point are hole we do the same as before we search the closer point 
    # that is a point of the signal
    if signal[-1] == 0:
        j = 1
        while signal[-j] == 0:
            j += 1
        signal[-1] = signal[-j]

    # If a point inside the signal is a hole we made the mean between the closer points 
    # before and after it which are in the signal
    for i in range(len(signal)-1): 
        if signal[i] == 0:
            a = i+1
            b = i-1
            while signal[a] == 0:
                a += 1
            while signal[b] == 0:
                b -= 1
            signal[i] = np.mean([signal[b],signal[a]])     
            
    return(signal[:end])


def lead_extraction(dic_tracks, extraction_method, TYPE, NOISE, DEBUG = False):
    
    """
    Extract the digital information from images
    
    Parameters
    ----------
    dic_tracks: dictionary, dictionary of track images
    extraction_method: str, it can be : "lazy", "full", "fragmented"
    TYPE  : str, format of the image
    NOISE : bool, if the image is noised or not
    DEBUG : bool, show the image
    
    Returns
    -------
    dictionary: dictionary of digital tracks
    """
    
    # Digital tracks dictionary 
    dic_extracted_tracks = {}
    dic_image_bin = {}
    dic_extracted_track_not_scale = {}
    for d in dic_tracks:
        # Kardia Files are already binarize
        image_bin = dic_tracks[d]
        # Plot the binarized image
        if DEBUG == True:
            plt.imshow(image_bin)
            plt.show()
            plt.imshow(image_bin)
            
                       
        
        # List of the extracted signal
        extraction = []
        
        if extraction_method == "lazy":
            extraction = lazy_extraction(image_bin)
        elif extraction_method == "full":
            extraction = full_extraction(image_bin)
        elif extraction_method == "fragmented":
            extraction = fragmented_extraction(image_bin)    
        # Removing the holes in the signal
        signal = sup_holes(extraction, TYPE)
        
        # Scale the signal in time each tracks length 10sec with a frequency of 500hz it is 5000pts
        # by tracks plus the reference pulse
        x = [i for i in range(len(signal))]
        y = signal
        if TYPE.lower() == 'classic':
            new_x = [i for i in np.arange(0,len(signal),len(signal)/5140)]
        elif TYPE.lower() == 'kardia':
            new_x = [i for i in np.arange(0,len(signal),len(signal)/4000)]
        else: 
            new_x = [i for i in np.arange(0,len(signal),len(signal)/5000)]
        signal_scale = np.interp(new_x,x,y) 
        dic_extracted_track_not_scale[d] = signal
        dic_extracted_tracks[d] = signal_scale
        dic_image_bin[d] = image_bin
        
        
        # Plot the signal before its scale
        if DEBUG == True:
            plt.plot(signal, c = 'r')
            plt.show()    
            
    return(dic_extracted_tracks, dic_image_bin, dic_extracted_track_not_scale)



def lead_cutting(dic_tracks, DPI, TYPE, FORMAT, page, NOISE, DEBUG):
    """
    Cut each tracks into leads
    
    Parameters
    ----------
    dic_tracks: dictionary, dictionary of track images
    DPI   : int, resolution
    TYPE  : str, format of the image
    NOISE : bool, if the image is noised or not
    DEBUG : bool, show the image
    
    Returns
    -------
    dictionary: dictionary of leads
    """
    # Dictionary with reference pulse for each tracks
    dic_ref_pulse    = {} 
    # Dictionary with the lead   
    dic_leads       = {} 
    LEAD_LENGTH = 0
    LEAD_NUMBER = 1
    dic_association = {0 : "II"}
    # If the it is a classical format
    if TYPE.lower() == 'classic' or (TYPE.lower() == 'kardia' and FORMAT == 'multilead'):
        if TYPE.lower() != 'classic':
            if page == 0:
                # the reference pulse lasts 0.28sec
                LENGTH_PULSE       = 240  
            else: 
                LENGTH_PULSE       = 0
            
        # The disposition of the ECG is 4x4
        if len(dic_tracks) == 4:
            # leads lasts 2.5sec if there are 4 tracks
            LEAD_NUMBER = 4 
            dic_association = {0 : ['I', 'AVR', 'V1', 'V4'],
                              1 : ['II', 'AVL', 'V2', 'V5'],
                              2 : ['III', 'AVF', 'V3', 'V6'],
                              3 : ['II']}
            dic_time = {'I':(0,1250), 'II':(0,1250),'III':(0,1250),
                       'AVR': (1250,2500), 'AVL': (1250,2500),'AVF': (1250,2500),
                       'V1': (2500,3750), 'V2': (2500,3750),'V3': (2500,3750),
                       'V4': (3750,5000), 'V5': (3750,5000),'V6': (3750,5000),
                       'IIc': (0,5000)}
        # The disposition of the ECG is 6x2
        elif len(dic_tracks) == 6: 
            # leads last 5sec if there are 6 tracks
            LEAD_NUMBER = 2      
            dic_association = {0 : ['I', 'V1'],
                              1 : ['II', 'V2'],
                              2 : ['III', 'V3'],
                              3 : ['AVR', 'V4'],
                              4 : ['AVL', 'V5'],
                              5 : ['AVF', 'V6'],}
            dic_time = {'I':(0,2500), 'II':(0,2500),'III':(0,2500),'AVR': (0,2500), 'AVL': (0,2500),'AVF': (0,2500),
                       'V1': (2500,5000), 'V2': (2500,5000),'V3': (2500,5000),'V4': (2500,5000), 'V5': (2500,5000),'V6': (2500,5000)}
            
        ########## METTRE UN ELSE ICI ##################    
        #else:
        ########## METTRE UN ELSE ICI ##################    
        

        if TYPE.lower() == 'kardia' and FORMAT.lower() == 'multilead':
            # leads last 10sec in kardia
            
            dic_association = {
            0 : "I",
            1 : "II",
            2 : "III",
            3 : "AVR",
            4 : "AVL",
            5 : "AVF",}
        
        # Plot each tracks
        for t in dic_tracks:
            if DEBUG == True:
                LENGTH_PULSE = 140
                print("Track :", t)
                plt.figure(figsize = (20,14))
                plt.plot(dic_tracks[t])     
                plt.axvline(LENGTH_PULSE, c = 'r')

            if TYPE.lower() != 'kardia':
                # Isolate the reference pulse
                LENGTH_PULSE = 330
                if len(dic_tracks) == 4:
                    LENGTH_PULSE = len(dic_tracks[t]) - 5000
                    
                elif len(dic_tracks) == 6:
                    LENGTH_PULSE = len(dic_tracks[t]) - 5000
                dic_ref_pulse[t] = dic_tracks[t][ : LENGTH_PULSE ] 
                
                # Pixel of amplitude 0mV
                pixel_zero = max(dic_ref_pulse[t])
                # Pixel of amplitude 1mV
                pixel_one  = min(dic_ref_pulse[t]) 
                # Define the factor
                f = pixel_zero - pixel_one 
                if f == 0:
                    f = 1

                
                # Define the beggining of lead part
                LEAD_LENGTH = int(len(dic_tracks[t][ LENGTH_PULSE:  ]) / LEAD_NUMBER)
                length = LENGTH_PULSE
                #length = LENGTH_PULSE  
                # Define the lead position
                it = 0                 

                # special case on the disposition 4x4 the last track containe 10sec of the lead II
                if len(dic_tracks) == 4 and t == 3: 
                    dic_leads['IIc'] = (((pixel_zero - dic_tracks[t][LENGTH_PULSE: 4 * LEAD_LENGTH+LENGTH_PULSE])/f) * 1000)
                    if DEBUG == True:
                        plt.show()

                # extract each lead from the tracks
                elif LEAD_LENGTH != 0 :
                    while length < len(dic_tracks[1]):
                        try :
                            dic_leads[dic_association[t][it]] = (((pixel_zero - dic_tracks[t][length : length + LEAD_LENGTH]) / f) * 1000) # We fill the leads dictionnary with the name of the lead and the image of it
                            length += int(len(dic_tracks[t][ LENGTH_PULSE:  ]) / LEAD_NUMBER)
                            it     += 1
                            if DEBUG == True:
                                plt.axvline(length, c = 'r')
                        except Exception as e:
                            length += int(len(dic_tracks[t][ LENGTH_PULSE:  ]) / LEAD_NUMBER)
                else :
                    return(0)
                if DEBUG == True:
                    plt.show()
       
            else:
                if page == 0:
                    ref_pulse = dic_tracks[t][ : LENGTH_PULSE ] 
                    # Pixel of amplitude 0mV
                    pixel_zero = max(ref_pulse)
                    # Pixel of amplitude 1mV
                    pixel_one  = min(ref_pulse) 
                    # Define the factor
                    f = pixel_zero - pixel_one 
                    if f == 0:
                        f = 1
                    # Define the beggining of lead part
                    length = LENGTH_PULSE 
                    dic_leads['ref'] = [pixel_zero,f]
                    
                    # Scale the signal in amplitude
                    dic_leads[dic_association[t]] = ((pixel_zero - dic_tracks[t][length:]) / f)* 1000
                
                else:
                    length = 0
                    dic_leads[dic_association[t]] = dic_tracks[t][length:]
          
        try:
            for k in dic_leads:
                zero_vector = np.zeros(5000)
                zero_vector[dic_time[k][0]:dic_time[k][1]] = dic_leads[k]
                dic_leads[k] = zero_vector
        except Exception as e:
            pass
        return(dic_leads)
    
    # If the format is not classic
    else:
        if TYPE.lower() == 'apple':
            # the reference pulse lasts 0.28sec
            LENGTH_PULSE       = 180  
        elif TYPE.lower() == 'kardia':
            LENGTH_PULSE       = 240
        else:
            # the reference pulse lasts 0.28sec
            LENGTH_PULSE       = 300   
        
        
        for t in dic_tracks:
            if t == 0 :
                # Plot each tracks
                if DEBUG == True:
                    plt.figure(figsize = (20,14))
                    plt.plot(dic_tracks[t])     
                    plt.axvline(LENGTH_PULSE, c = 'r')
                    plt.show()
                
                # Isolate the reference pulse
                dic_ref_pulse = dic_tracks[t][ : LENGTH_PULSE ]
                # Pixel of amplitude 0mV
                pixel_zero = max(dic_ref_pulse[:10])
                # Pixel of amplitude 1mV
                pixel_one  = min(dic_ref_pulse) 
                
                # if we have not extracted the reference pulse
                if np.all(np.diff(dic_ref_pulse) == 0): 
                    pixel_zero = np.mean(dic_ref_pulse)
                    pixel_one  = min(dic_ref_pulse)
                    
                # Define the factor
                f = pixel_zero - pixel_one 
                if f == 0:
                    f = pixel_zero - (pixel_zero - ((10*DPI)/25.4))
                
                # Separate the signal from the reference pulse
                all_signal = dic_tracks[t][LENGTH_PULSE : ]
                
            # Concatane the signal if it is on more than one track   
            else:
                dist = np.mean(all_signal) - np.mean(dic_tracks[t])
                all_signal = np.concatenate((all_signal,dic_tracks[t]+dist), axis = 0)
            
            # Plot the different pixel  
            if DEBUG == True:
                print("0 : ", pixel_zero)
                print("1 : ", pixel_one)
                print("1st pixel : ", all_signal[0])
        
        # Scale the signal in amplitude
        new_signal = np.zeros((len(all_signal)))
        for v in range(len(all_signal)):
            new_signal[v] = ((pixel_zero-all_signal[v])/f)*1000 # Scale the point in function of the Zero pixel and the
        
        return(new_signal )
