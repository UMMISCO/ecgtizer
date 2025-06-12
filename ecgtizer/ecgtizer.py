#Modules 
from .PDF2XML import convert_PDF2image, check_noise_type, text_extraction, tracks_extraction, clean_tracks, sup_holes, lead_extraction, lead_cutting
from .PDF2XML_mod import plot_function, write_xml, plot_overlay
from .completion import completion_
import cv2


import numpy as np
import time 

class ECGtizer:
    """
    Class permettant de convertir les ECGs au format PDF vers un format XML
    input : - str : File name
            - str : PDF format (optional)
            - int :  Density Per Inch (DPI) (optionel)
            - bool : Verbose to describe each step
    output : - array: Digitize leads
    
    """

    def __init__(self, file, dpi, Callback = None, extraction_method = "full",typ = "", verbose = False, DEBUG = False):
        ### Variables ###
        self.file = file
        self.typ  = typ
        self.dpi  = dpi
        self.good = True
        self.extraction_method = extraction_method
        
        ### "Constant" ###
        self.page = 1
        self.extracted_lead = np.zeros((1,)) 
        
        self.table_parameters = {
                 'hour'              : 'unknow' ,
                 'day'               : 'unknow' , 
                 'month'             : 'unknow' , 
                 'year'              : 'unknow' ,
                 'Scale_x'           : '10'     ,
                 'Scale_y'           : '25'     ,
                 'low_freq'          : 'unknow' ,
                 'high_freq'         : 'unknow' ,
                 'BPM'               : 'unknow' ,
                 'Inter PR (ms)'     : 'unknow' ,
                 'Dur.QRS (ms)'      : 'unknow' ,
                 'QT (ms)'           : 'unknow' ,
                 'QTc (ms)'          : 'unknow' ,
                 'Axe P'             : 'unknow' ,
                 'Axe R'             : 'unknow' ,
                 'Axe T'             : 'unknow' ,
                 'Moy RR (ms)'       : 'unknow' ,
                 'QTcB (ms)'         : 'unknow' ,
                 'QTcF (ms)'         : 'unknow' ,
                 'Rythme'            : 'unknow' ,
                 'ECG'               : 'unknow' ,
                 'Age'               : 'unknow' ,
                 'sex'               : 'unknow' ,
                 'other_information' : 'unknow' ,    
        }
        
        ### All dictionary where pre-processing images are stored ###
        self.all_image               = []
        self.all_image_clean         = []
        self.dic_tracks              = []
        self.dic_tracks_clean        = []
        self.dic_tracks_ex           = []
        self.dic_image_bin           = []
        self.df_patient              = []
        self.dic_tracks_ex_not_scale = []
        self.variance                = []
        
     
        
        ### Convert PDF files to image ###
        if file[-3:] == 'pdf':
            if verbose == True:
                print("\n")
                print("--- Conversion PDF in image : ", end='')
                start = time.time()
            if Callback != None:
                Callback("\n")
                Callback("--- Conversion PDF in image : ", end='')
                start = time.time()
            images, page_number, _ = convert_PDF2image(file, DPI = dpi)
            if _ == False:
                self.good = False
                return(None)
            self.all_image = images
            if verbose == True:
                print("\t\t\tOK ("+str(round(start - time.time(), 2)) + "sec) \n")
            if Callback != None:
                Callback("\t\t\tOK ("+str(round(start - time.time(), 2)) + "sec) \n")
            page = 0
        elif file[-3:] == 'png' or file[-3:] == 'jpg' or file[-4:] == 'jpeg':
            if verbose == True:
                print("\n")
                print("--- Open Image : ", end='')
                start = time.time()
            images = [cv2.imread(file)]
            page = 0
            if verbose == True:
                print("\t\t\tOK ("+str(round(start - time.time(), 2)) + "sec) \n")
        
        ### Convert all images ###
        for image in images:
            if page > 0 and verbose == True:
                print("\n\n")
            
            ### a/ Check the image type ###
            if verbose == True:
                print("--- Check Quality and Type of image : ", end='')
                start = time.time()
            if Callback != None:
                Callback("--- Check Quality and Type of image : ", end='')
                start = time.time()
            self.image = np.array(image)
            TYPE, NOISE = check_noise_type(self.image, dpi, DEBUG) 
            if self.typ != "":
                TYPE = self.typ
            # Kardia Format is particular
            if TYPE.lower() == 'kardia':
                if page_number > 1:
                    FORMAT = 'multilead'
                else:
                    FORMAT = 'unilead'
            else: 
                FORMAT = ''
            if verbose == True:
                print("\t\tOK ("+str(round(start - time.time(), 2)) + "sec) \n")
            if Callback != None:
                Callback("\t\tOK ("+str(round(start - time.time(), 2)) + "sec) \n")
                
            ### b/ Check the image type ### 
            if verbose == True:
                print("TYPE", TYPE)
                print("--- Extract all the text from the image : ", end='')
                start = time.time()
            if Callback != None:
                Callback("--- Extract all the text from the image : ", end='')
                start = time.time()
            #image_clean, df = text_extraction(self.image,page, dpi, NOISE, TYPE, DEBUG = DEBUG)
            image_clean = text_extraction(self.image,page, dpi, NOISE, TYPE, DEBUG = DEBUG)
            #if page == 0:
            #    self.df_patient = df
            image = image_clean
            if verbose == True:
                print("\tOK ("+str(round(start - time.time(), 2)) + "sec) \n")
            if Callback != None:
                Callback("\tOK ("+str(round(start - time.time(), 2)) + "sec) \n")
            
            ### c/ Extract each tracks ### 
            if verbose == True:
                print("--- Detect tracks position : ", end='')
                start = time.time()
            if Callback != None:
                Callback("--- Detect tracks position : ", end='')
                start = time.time()
            dic_tracks, varianceh, variancev = tracks_extraction(self.image, TYPE, dpi, FORMAT, DEBUG = DEBUG, NOISE = NOISE)
            self.varianceh = varianceh
            self.variancev = variancev
            self.dic_tracks = dic_tracks
            if verbose == True:
                print("\t\t\tOK ("+str(round(start - time.time(), 2)) + "sec) \n")
            if Callback != None:
                Callback("\t\t\tOK ("+str(round(start - time.time(), 2)) + "sec) \n")
            
            ### c/ Clean the tracks from outliers ### 
            #if verbose == True:
            #    print("--- Clean tracks : ", end='')
            #    start = time.time()
            #if TYPE.lower() != 'kardia':
            #    clean_tracks(dic_tracks,TYPE, NOISE = NOISE, DEBUG = DEBUG )
            #    self.dic_tracks_clean = clean_tracks
            #if verbose == True:
            #    print("\t\t\t\tOK ("+str(round(start - time.time(), 2)) + "sec) \n")

            
            ### d/ Convert the tracks in digital format ### 
            if verbose == True:
                print("--- Tracks extraction : ", end='')
                start = time.time()
            if Callback != None:
                Callback("--- Tracks extraction : ", end='')
                start = time.time()
            dic_tracks_ex, image_bin, dic_tracks_ex_not_scale = lead_extraction(dic_tracks, extraction_method, TYPE, NOISE = NOISE, DEBUG = DEBUG )
            #dic_tracks_ex = lead_extraction(dic_tracks,TYPE, NOISE = NOISE, DEBUG = DEBUG )
            self.dic_tracks_ex = dic_tracks_ex
            self.dic_tracks_ex_not_scale = dic_tracks_ex_not_scale
            if verbose == True:
                print("\t\t\tOK ("+str(round(start - time.time(), 2)) + "sec) \n")
            if Callback != None:
                Callback("\t\t\tOK ("+str(round(start - time.time(), 2)) + "sec) \n")
                
                
                
            if verbose == True:
                print("--- Lead detection : ", end='')
                start = time.time()    
            if Callback != None:
                Callback("--- Lead detection : ", end='')
                start = time.time()
            dic_lead = lead_cutting(dic_tracks_ex, dpi,TYPE, FORMAT, page, NOISE = NOISE,  DEBUG = DEBUG )
            if verbose == True:
                print("\t\t\t\tOK ("+str(round(start - time.time(), 2)) + "sec) \n")
            if Callback != None:
                Callback("\t\t\t\tOK ("+str(round(start - time.time(), 2)) + "sec) \n")


            
                
            if TYPE.lower() == 'kardia' and FORMAT.lower() == 'multilead':
                if page == 0:
                    extracted_lead = dic_lead
                else:
                    for k in dic_lead:
                        pixel_zero = extracted_lead['ref'][0]
                        f = extracted_lead['ref'][1]
                        dic_lead[k] = ((pixel_zero - dic_lead[k]) / f) * 1000
                        extracted_lead[k] = np.concatenate([extracted_lead[k], dic_lead[k]])
                page +=1
                
            elif page == 0 and TYPE.lower() != 'classic' and  FORMAT.lower() != 'multilead':
                extracted_lead = {}
                extracted_lead["all"] = dic_lead
                page +=1
                
            elif TYPE.lower() != 'classic':
                extracted_lead["all"] = np.concatenate((np.array(extracted_lead["all"]),np.array(dic_lead)))
                page +=1
            
            self.dic_image_bin.append(image_bin)

        if TYPE.lower() == 'classic':
            self.extracted_lead = dic_lead
        else:
            self.extracted_lead = extracted_lead
        
        self.dic_tracks = dic_tracks
        self.TYPE = TYPE
                
    ### Plot the signal Extracted ###
    
    def plot(self,lead = "",  begin = 0, end = 'inf', c = None, save = False, transparent = False, completion = False):
        if completion == False:
            plot_function(lead_all = self.extracted_lead, lead = lead, b = begin, e = end, c = c , save = save, transparent=transparent)
        else:
            plot_function(lead_all = self.extracted_lead_comp, lead = lead, b = begin, e = end, c = c , save = save, transparent=transparent)
    
    def plot_over(self):
        plot_overlay(lead = self.dic_tracks_ex_not_scale, image = self.image, piqueh = self.varianceh, piquev = self.variancev)
        
    ### Save the ecg on xml ###
    def save_xml (self, save, num_version = '0.0', date_version = "17.O4.2023"):
        write_xml(matrix = self.extracted_lead, path_out = save, TYPE = self.TYPE, table = self.table_parameters, 
                  num_version = num_version, date_version  = date_version)
        
    def completion(self, path_model, device):
        self.extracted_lead_comp = completion_(ecg = self.extracted_lead, path_model = path_model, device = device)
        
        
        
