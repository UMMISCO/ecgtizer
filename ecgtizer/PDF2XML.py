from __future__ import annotations

import logging

import numpy as np
from pdf2image import convert_from_path, exceptions
from .extraction_functions import lazy_extraction, full_extraction, fragmented_extraction
import cv2
from scipy import signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# --- Signal parameters ---
SAMPLING_FREQ = 500          # Hz
SIGNAL_LENGTH_STANDARD = 5000  # 10 seconds at 500 Hz (Wellue/other)
SIGNAL_LENGTH_CLASSIC = 5140   # Classic format signal length
SIGNAL_LENGTH_KARDIA = 4000    # Kardia format signal length
AMPLITUDE_SCALE_UV = 1000     # Scaling factor for µV conversion

# --- Reference pulse lengths (in samples) ---
REF_PULSE_APPLE = 180
REF_PULSE_KARDIA = 240
REF_PULSE_GENERIC = 300
REF_PULSE_CLASSIC = 330

# --- Image noise/variance thresholds ---
VARIANCE_NOISY = 3000         # Above this → image is noisy
VARIANCE_HIGH = 2000          # Above this or below LOW → might be noisy
VARIANCE_LOW = 600            # Below HIGH or above this → might be noisy
NOISE_PARTIAL = 0.5           # Intermediate noise state

# --- Image processing thresholds ---
LINE_VARIANCE_MIN = 1000      # Min row variance to keep during image cleanup
COLUMN_VARIANCE_MIN = 200     # Min column variance to keep during image cleanup
WAVEFORM_VARIANCE_MIN = 200   # Min vertical variance to detect signal presence

# --- Pixel values ---
WHITE_PIXEL = 255

# --- Lead timing boundaries (samples) ---
LEAD_TIME_3X4 = {
    'I': (0, 1250), 'II': (0, 1250), 'III': (0, 1250),
    'AVR': (1250, 2500), 'AVL': (1250, 2500), 'AVF': (1250, 2500),
    'V1': (2500, 3750), 'V2': (2500, 3750), 'V3': (2500, 3750),
    'V4': (3750, 5000), 'V5': (3750, 5000), 'V6': (3750, 5000),
    'IIc': (0, 5000),
}
LEAD_TIME_6X2 = {
    'I': (0, 2500), 'II': (0, 2500), 'III': (0, 2500),
    'AVR': (0, 2500), 'AVL': (0, 2500), 'AVF': (0, 2500),
    'V1': (2500, 5000), 'V2': (2500, 5000), 'V3': (2500, 5000),
    'V4': (2500, 5000), 'V5': (2500, 5000), 'V6': (2500, 5000),
}




def _binarize_image(image: np.ndarray, TYPE: str, NOISE: bool | float) -> np.ndarray:
    """Binarize a grayscale or BGR image using the appropriate thresholding method."""
    if image.ndim == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    if NOISE:
        _, image_bin = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY_INV)
    elif TYPE.lower() == "wellue":
        _, image_bin = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        _, image_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return image_bin


def _calibrate_ref_pulse(ref_pulse: np.ndarray, DPI: int = 0) -> tuple[float, float]:
    """Compute calibration (pixel_zero, factor) from a reference pulse segment.

    Returns (pixel_zero, factor) where factor converts pixel distance to µV.
    """
    pixel_zero = float(max(ref_pulse[:10]) if len(ref_pulse) >= 10 else max(ref_pulse))
    pixel_one = float(min(ref_pulse))

    # Flat pulse fallback
    if np.all(np.diff(ref_pulse) == 0):
        pixel_zero = float(np.mean(ref_pulse))
        pixel_one = float(min(ref_pulse))

    f = pixel_zero - pixel_one
    if f == 0:
        if DPI > 0:
            f = (10 * DPI) / 25.4
        else:
            f = 1.0
    return pixel_zero, f


def convert_PDF2image(path_input: str, DPI: int) -> np.ndarray:
    
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
        logger.error("Impossible conversion. The input file is not a PDF.")
        return("_", "_", False)
    return(pages,len(pages), True)



def check_noise_type(image: np.ndarray, DPI: int, DEBUG: bool) -> tuple[str, bool | float]:
    
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
    if np.var(image) > VARIANCE_HIGH or np.var(image) < VARIANCE_LOW:
        if np.var(image) > VARIANCE_NOISY:
            NOISE = True
        else:
            NOISE = NOISE_PARTIAL
    
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
        if DEBUG:
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
        
        
def text_extraction(image: np.ndarray, page: int, DPI: int, NOISE: bool | float, TYPE: str, DEBUG: bool) -> tuple[dict, np.ndarray, str, str]:
    
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
    df = []
    
    if TYPE.lower() == 'kardia':
        # Isolate the record region
        work_image = np.array(image)[DPI:int(10*DPI),int(0.3*DPI):int(8*DPI)]
        # Convert the image in gray scale
        image_gray = cv2.cvtColor(work_image,cv2.COLOR_BGR2GRAY) 
        # Binarize the image thanks to the gray scale
        new_image = np.where(image_gray == 0, WHITE_PIXEL, 0).astype(image_gray.dtype)
        


        # Compute the vertical variance
        var_line = np.var(new_image, axis= 1)
        # Compute the horizontal variance
        var_column = np.var(new_image, axis = 0)
        # Define a second image to work with
        working_image = np.copy(new_image)

        
        
        for i in range(len(new_image)):
            if var_line[i] < LINE_VARIANCE_MIN:
                working_image[i,:] = 0
        for i in range(len(new_image[0])):
            if var_column[i] < COLUMN_VARIANCE_MIN:
                working_image[:,i] = 0
        
        if DEBUG:
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
    if NOISE:
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
        
    # Plot the image with the detected rectangles
    if DEBUG:
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



def tracks_extraction(image: np.ndarray, TYPE: str, DPI: int, FORMAT: str, NOISE: bool | float = False, DEBUG: bool = False) -> dict[int, np.ndarray]:
    
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
            if DEBUG:
                for im in dic_tracks_temp:
                    plt.imshow(dic_tracks_temp[im])
                    plt.show()
            return(dic_tracks_temp)
        
        else: 
            if DEBUG:
                for im in dic_tracks:
                    plt.imshow(dic_tracks[im])
                    plt.show()
            return(dic_tracks)
    
    # Plot the original image 
    if DEBUG:
        plt.figure(figsize = (20,14))
    
    # Binarize the image using the appropriate thresholding method
    image_bin = _binarize_image(image, TYPE, NOISE)
    

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
        # if NOISE:
        #     peaksh, _ = find_peaks(horizontal_variance, height=(len(image)-(len(image[0])*15/100),len(image[0])), distance=int(len(image)/10))
        

    
    if DEBUG:
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
        
        if DEBUG:
            plt.axhline(cut_pos[c], c = 'g', alpha = 0.6)
            
    # Plot the position of the cut in the image 
    if DEBUG:
        plt.imshow(image)
        plt.axhline(cut_pos[-1], c = 'g', alpha = 0.6)
        for p in peaksh:
            plt.axhline(p, c = 'r', alpha = 0.6)
        
    # Compute the vertical variance   
    vertical_variance = np.var(image_bin, axis = 0) 

    # Find positions where variance indicates signal waveform presence
    peaksv = np.where(vertical_variance > WAVEFORM_VARIANCE_MIN)[0].tolist()
    
    
            
    # For all the tracks we cut vertically the part which not contain waveform
    for track in dic_tracks.keys(): 
        dic_tracks[track] = dic_tracks[track][:,peaksv[0]:peaksv[-1]]
    
    # Plot the position of the cut in the image 
    if DEBUG:
        plt.axvline(peaksv[0])
        plt.axvline(peaksv[-1])
        plt.savefig("Image_of_tracks.png")
        plt.show()
        
    if DEBUG:
        plt.plot(vertical_variance)
        
        
        plt.axvline(peaksv[0], c = "r")
        plt.axvline(peaksv[-1], c = "r")
        plt.savefig("Vertical_variance.png")
        plt.show()
    
    
    return(dic_tracks, peaksh, peaksv[0])




def clean_tracks(dic_tracks: dict[int, np.ndarray], TYPE: str, NOISE: bool | float, DEBUG: bool) -> dict[int, np.ndarray]:
    
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
                    image_bin = np.where(image_bin, WHITE_PIXEL, 0).astype(np.uint8)

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
        if DEBUG:
            plt.figure(figsize = (20,14))
            try:
                plt.imshow(rect)
            except UnboundLocalError:
                plt.imshow(im2)
            plt.show()
        
def sup_holes(signal: list | np.ndarray, TYPE: str) -> np.ndarray:
    
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

    # Interpolate interior holes (zeros) using nearest non-zero neighbours
    signal = np.asarray(signal, dtype=float)
    zero_mask = signal == 0
    if np.any(zero_mask):
        nonzero_idx = np.where(~zero_mask)[0]
        if len(nonzero_idx) > 0:
            signal[zero_mask] = np.interp(
                np.where(zero_mask)[0], nonzero_idx, signal[nonzero_idx]
            )     
            
    return(signal[:end])


def lead_extraction(dic_tracks: dict[int, np.ndarray], extraction_method: str, TYPE: str, NOISE: bool | float, DEBUG: bool = False) -> dict[str, np.ndarray]:
    
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
        if DEBUG:
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
            new_x = [i for i in np.arange(0,len(signal),len(signal)/SIGNAL_LENGTH_CLASSIC)]
        elif TYPE.lower() == 'kardia':
            new_x = [i for i in np.arange(0,len(signal),len(signal)/SIGNAL_LENGTH_KARDIA)]
        else:
            new_x = [i for i in np.arange(0,len(signal),len(signal)/SIGNAL_LENGTH_STANDARD)]
        signal_scale = np.interp(new_x,x,y) 
        dic_extracted_track_not_scale[d] = signal
        dic_extracted_tracks[d] = signal_scale
        dic_image_bin[d] = image_bin
        
        
        # Plot the signal before its scale
        if DEBUG:
            plt.plot(signal, c = 'r')
            plt.show()    
            
    return(dic_extracted_tracks, dic_image_bin, dic_extracted_track_not_scale)



def lead_cutting(dic_tracks: dict[int, np.ndarray], DPI: int, TYPE: str, FORMAT: str, page: int, NOISE: bool | float, DEBUG: bool) -> dict[str, np.ndarray] | np.ndarray:
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
                LENGTH_PULSE       = REF_PULSE_KARDIA
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
            dic_time = LEAD_TIME_3X4
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
            dic_time = LEAD_TIME_6X2
            
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
            if DEBUG:
                LENGTH_PULSE = 140
                logger.debug("Track: %s", t)
                plt.figure(figsize = (20,14))
                plt.plot(dic_tracks[t])     
                plt.axvline(LENGTH_PULSE, c = 'r')

            if TYPE.lower() != 'kardia':
                # Isolate the reference pulse
                LENGTH_PULSE = REF_PULSE_CLASSIC
                if len(dic_tracks) == 4:
                    LENGTH_PULSE = len(dic_tracks[t]) - SIGNAL_LENGTH_STANDARD

                elif len(dic_tracks) == 6:
                    LENGTH_PULSE = len(dic_tracks[t]) - SIGNAL_LENGTH_STANDARD
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
                    dic_leads['IIc'] = (((pixel_zero - dic_tracks[t][LENGTH_PULSE: 4 * LEAD_LENGTH+LENGTH_PULSE])/f) * AMPLITUDE_SCALE_UV)
                    if DEBUG:
                        plt.show()

                # extract each lead from the tracks
                elif LEAD_LENGTH != 0 :
                    while length < len(dic_tracks[1]):
                        try :
                            dic_leads[dic_association[t][it]] = (((pixel_zero - dic_tracks[t][length : length + LEAD_LENGTH]) / f) * AMPLITUDE_SCALE_UV) # We fill the leads dictionnary with the name of the lead and the image of it
                            length += int(len(dic_tracks[t][ LENGTH_PULSE:  ]) / LEAD_NUMBER)
                            it     += 1
                            if DEBUG:
                                plt.axvline(length, c = 'r')
                        except Exception as e:
                            length += int(len(dic_tracks[t][ LENGTH_PULSE:  ]) / LEAD_NUMBER)
                else :
                    return(0)
                if DEBUG:
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
                    dic_leads[dic_association[t]] = ((pixel_zero - dic_tracks[t][length:]) / f) * AMPLITUDE_SCALE_UV
                
                else:
                    length = 0
                    dic_leads[dic_association[t]] = dic_tracks[t][length:]
          
        try:
            for k in dic_leads:
                zero_vector = np.zeros(SIGNAL_LENGTH_STANDARD)
                zero_vector[dic_time[k][0]:dic_time[k][1]] = dic_leads[k]
                dic_leads[k] = zero_vector
        except Exception as e:
            pass
        return(dic_leads)
    
    # If the format is not classic
    else:
        if TYPE.lower() == 'apple':
            LENGTH_PULSE       = REF_PULSE_APPLE
        elif TYPE.lower() == 'kardia':
            LENGTH_PULSE       = REF_PULSE_KARDIA
        else:
            LENGTH_PULSE       = REF_PULSE_GENERIC   
        
        
        for t in dic_tracks:
            if t == 0 :
                # Plot each tracks
                if DEBUG:
                    plt.figure(figsize = (20,14))
                    plt.plot(dic_tracks[t])     
                    plt.axvline(LENGTH_PULSE, c = 'r')
                    plt.show()
                
                # Isolate and calibrate the reference pulse
                dic_ref_pulse = dic_tracks[t][ : LENGTH_PULSE ]
                pixel_zero, f = _calibrate_ref_pulse(dic_ref_pulse, DPI)
                
                # Separate the signal from the reference pulse
                all_signal = dic_tracks[t][LENGTH_PULSE : ]
                
            # Concatane the signal if it is on more than one track   
            else:
                dist = np.mean(all_signal) - np.mean(dic_tracks[t])
                all_signal = np.concatenate((all_signal,dic_tracks[t]+dist), axis = 0)
            
            # Plot the different pixel  
            if DEBUG:
                logger.debug("0: %s", pixel_zero)
                logger.debug("1: %s", pixel_one)
                logger.debug("1st pixel: %s", all_signal[0])
        
        # Scale the signal in amplitude
        new_signal = ((pixel_zero - all_signal) / f) * AMPLITUDE_SCALE_UV

        return(new_signal)
