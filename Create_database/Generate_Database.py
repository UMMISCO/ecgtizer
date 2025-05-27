import sys
sys.path.append("../../ecgtizer_old/ecgtizer/ecgtizer/")
from XML2PDF import Write_PDF
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json


from ecg_image_generator.CreasesWrinkles.creases import get_creased
from concurrent.futures import ProcessPoolExecutor


import wfdb
import ast
from scipy.signal import find_peaks
from random import random

from tqdm import tqdm

import numpy as np


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def array_to_dict(data):
    dic_lead = {0:"I", 1:"II", 2:"III", 3:"AVR", 4:"AVL", 5:"AVF", 6:"V1", 7:"V2", 8:"V3", 9:"V4", 10:"V5", 11:"V6"}
    new_data = {}
    for i in range(12):
        new_data[dic_lead[i]] = data[:,i]*1000
    return new_data


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
    # Plot the original image 
    # Convert the image in gray scale 
    img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian Blur
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    
    # If the image is noised we will use the Sauvola detection thresholding
    
    # If the image is Wellue type we have determine the optimal threshold
    
    # If the image is not noised we will use the Otsu thresholding    
    ret,image_bin = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
    
    # Compute the horizontal variance on binarized image
    horizontal_variance     = np.var(image_bin, axis = 1) 
    
    
    # If images are taller than they are wide the distance between two peaks is smaller
    if len(image) > len(image[0]):
        # Compute the pikes position
        peaks = signal.argrelextrema(horizontal_variance, np.greater, order = int(0.010*len(image)))[0] 
        
    # If images are wide than they are taller the distance between two peaks is bigger
    if len(image) < len(image[0]):
        # Compute the pikes position
        #peaks = signal.argrelextrema(horizontal_variance, np.greater, order = int(0.05*len(image)))[0] 
        peaks, _ = find_peaks(horizontal_variance, height=None, distance=int(len(image)/15))

    if DEBUG == True:
        plt.plot(horizontal_variance)
        for p in peaks:
            plt.axvline(p, c = "r")
        plt.savefig("Horrizontal_variance.png")
        plt.show()
        plt.figure(figsize = (20,20))
        
    
    # Define a list with all the position to cut between tracks a we store the beggining of the image
    cut_pos = [0]
    # for all peaks we only keep the position between them
    for p in range(len(peaks)-1):
        cut_pos.append(int((peaks[p]+peaks[p+1])/2))
    # We store the ending of the image
    cut_pos.append(len(image))


    new_cut_pos = []
    possible_overlapping = []
    # Better Crop the image
    for c in range(len(cut_pos)-1):
        horizontal_variance = np.var(image_bin[cut_pos[c]:cut_pos[c+1]], axis = 1)
       
        new_peaks = []
        for p in range(len(horizontal_variance)):
            if horizontal_variance[p] > 0:
                new_peaks.append(p)        
        # #plt.imshow(image[cut_pos[c]:cut_pos[c+1]])
        # plt.plot(horizontal_variance)
        # plt.axvline(peaks[0], c = "r")
        # plt.axvline(peaks[-1], c = "r")
        # plt.show()
        if new_peaks[0] == 0:
            possible_overlapping.append(True)
        else:
            possible_overlapping.append(False)
        new_cut_pos.append(new_peaks[0]+cut_pos[c])
        new_cut_pos.append(new_peaks[-1]+cut_pos[c])
    cut_pos = []
    for i in range(len(new_cut_pos)):
        if i % 2 == 0:
            cut_pos.append(new_cut_pos[i])
        else:
            cut_pos.append(new_cut_pos[i])
    

    
    # If we have 6 tracks we have extracted text information
    # if len(cut_pos) == 6:
    #     del cut_pos[0]
    
    # We store all track image in the dictionary 
    it = 1
    bounding_boxes_y = []
    bounding_boxes_height = []
    bounding_boxes_y_center = []
    for c in range(0,len(cut_pos)-1,2):
        if it == 1:
            bounding_boxes_y.append((cut_pos[c], cut_pos[c], cut_pos[c+1], cut_pos[c+1]))
            bounding_boxes_y_center.append(cut_pos[c]+((cut_pos[c+1]-cut_pos[c])/2))
            bounding_boxes_height.append(cut_pos[c+1]-cut_pos[c])
            dic_tracks[c] = image[cut_pos[c]+int(0.05*len(image)):cut_pos[c+1]]
        elif it == len(cut_pos)-1:
            bounding_boxes_y.append((cut_pos[c],cut_pos[c], cut_pos[c+1],  cut_pos[c+1]))
            bounding_boxes_y_center.append(cut_pos[c]+((cut_pos[c+1]-cut_pos[c])/2))
            bounding_boxes_height.append(cut_pos[c+1]-cut_pos[c])
            dic_tracks[c] = image[cut_pos[c]:cut_pos[c+1]]
        else:
            bounding_boxes_y.append((cut_pos[c], cut_pos[c], cut_pos[c+1], cut_pos[c+1]))
            bounding_boxes_y_center.append(cut_pos[c]+((cut_pos[c+1]-cut_pos[c])/2))
            bounding_boxes_height.append(cut_pos[c+1]-cut_pos[c])
            dic_tracks[c] = image[cut_pos[c]:cut_pos[c+1]]
            
        it+=1
        
        if DEBUG == True:
            
            plt.axhline(cut_pos[c], c = 'g', alpha = 0.6)
            
    # Plot the position of the cut in the image 
    if DEBUG == True:
        
        plt.imshow(image)
        plt.axhline(cut_pos[-1], c = 'g', alpha = 0.6)
        for p in peaks:
            plt.axhline(p, c = 'r', alpha = 0.6)
        
    # Compute the vertical variance   
    vertical_variance = np.var(image_bin, axis = 0) 
    # Define a list which will contain the pikes position
    peaks = [] 
    for var in range(len(vertical_variance)):
        # If the variance is no null then there is a signal waveform a we must not cut here the signal
        if vertical_variance[var] > 0 : 
            # Pikes take the beggining position of the waveform
            peaks.append(var)
    
    
    bounding_boxes_x = []
    bounding_boxes_x_center = []
    bounding_boxes_width = []
    # For all the tracks we cut vertically the part which not contain waveform
    for track in dic_tracks.keys(): 
        bounding_boxes_x.append((peaks[0], peaks[-1],peaks[0],peaks[-1]))
        bounding_boxes_x_center.append((peaks[-1]-peaks[0])/2)
        bounding_boxes_width.append(peaks[-1]-peaks[0])
        dic_tracks[track] = dic_tracks[track][:,peaks[0]:peaks[-1]]
    
    # Plot the position of the cut in the image 
    if DEBUG == True:
        plt.axvline(peaks[0])
        plt.axvline(peaks[-1])
        plt.savefig("Image_of_tracks.png")
        plt.show()
        
    if DEBUG == True:
        plt.plot(vertical_variance)
        
        
        plt.axvline(peaks[0], c = "r")
        plt.axvline(peaks[-1], c = "r")
        plt.savefig("Vertical_variance.png")
        plt.show()
    
    
    return(dic_tracks, bounding_boxes_x, bounding_boxes_y, bounding_boxes_x_center, bounding_boxes_y_center, bounding_boxes_width, bounding_boxes_height, possible_overlapping)

def rotate_image(seed,image, bounding_boxes_x, bounding_boxes_y, bounding_boxes_x_paper, bounding_boxes_y_paper):
    #np.random.seed(seed)
    #negative = np.random.choice([True, False])
    np.random.seed(seed)
    angle = np.random.choice([0, 90, 270])
    np.random.seed(seed)
    plus = np.random.randint(-30,30)
    #if negative:
        #rotation = angle - plus
    #else:
    rotation = angle + plus
    # Obtenir les dimensions de l'image
    (h, w) = image.shape[:2]
    # Calculer le centre de l'image originale
    (centerX, centerY) = (w // 2, h // 2)

    # Définir la matrice de rotation avec un angle de 90 degrés
    matrice_rotation = cv2.getRotationMatrix2D((centerX, centerY), rotation, 1.0)

    # Calculer les nouvelles dimensions pour s'assurer que l'image ne soit pas croppée
    cos = np.abs(matrice_rotation[0, 0])
    sin = np.abs(matrice_rotation[0, 1])
    nouveau_largeur = int((h * sin) + (w * cos))
    nouvelle_hauteur = int((h * cos) + (w * sin))

    # Ajuster la matrice de rotation pour tenir compte des nouvelles dimensions
    matrice_rotation[0, 2] += (nouveau_largeur / 2) - centerX
    matrice_rotation[1, 2] += (nouvelle_hauteur / 2) - centerY

    # Appliquer la rotation avec les nouvelles dimensions
    image_incline = cv2.warpAffine(image, matrice_rotation, (nouveau_largeur, nouvelle_hauteur), borderValue=(255, 255, 255))

    new_bounding_box_x = []
    new_bounding_box_y = []
    new_bounding_box_x_center = []
    new_bounding_box_y_center = []
    new_bounding_box_x_paper = []
    new_bounding_box_y_paper = []
    new_bounding_height = []
    new_bounding_width = []

    x0_paper = bounding_boxes_x_paper[0]
    y0_paper = bounding_boxes_y_paper[0]
    x1_paper = bounding_boxes_x_paper[1]
    y1_paper = bounding_boxes_y_paper[2]
    points_paper = np.array([
        [x0_paper, y0_paper],  # Haut gauche
        [x1_paper, y0_paper],  # Haut droit
        [x0_paper, y1_paper],  # Bas gauche
        [x1_paper, y1_paper],  # Bas droit
    ]) 
    points_paper_homogenes = np.hstack([points_paper, np.ones((4, 1))])
    points_paper_rotates = np.dot(matrice_rotation, points_paper_homogenes.T).T
    points_paper_rotates = points_paper_rotates.astype(int)

    new_bounding_box_x_paper = (points_paper_rotates[0][0],points_paper_rotates[1][0],points_paper_rotates[2][0],points_paper_rotates[3][0])
    new_bounding_box_y_paper = (points_paper_rotates[0][1],points_paper_rotates[1][1],points_paper_rotates[2][1],points_paper_rotates[3][1])
    
    ### Apply Rotation to the bounding boxes
    for i in range(len(bounding_boxes_x)):
        # Définir les 4 coins de la bounding box
        

        x0 = bounding_boxes_x[i][0]
        y0 = bounding_boxes_y[i][0]
        x1 = bounding_boxes_x[i][1]
        y1 = bounding_boxes_y[i][2]
        points = np.array([
            [x0, y0],  # Haut gauche
            [x1, y0],  # Haut droit
            [x0, y1],  # Bas gauche
            [x1, y1],  # Bas droit
        ])
        
        points_homogenes = np.hstack([points, np.ones((4, 1))])

        # Appliquer la matrice de rotation aux points
        points_rotates = np.dot(matrice_rotation, points_homogenes.T).T

        # Convertir les points rotatés en un format pour dessin
        points_rotates = points_rotates.astype(int)

        # Appliquer la matrice de rotation aux points
        new_bounding_box_x.append((points_rotates[0][0],points_rotates[1][0],points_rotates[2][0],points_rotates[3][0]))
        new_bounding_box_y.append((points_rotates[0][1],points_rotates[1][1],points_rotates[2][1],points_rotates[3][1]))
        new_bounding_box_x_center.append(((points_rotates[0][0] + points_rotates[1][0]) / 2))
        new_bounding_box_y_center.append((((points_rotates[0][1]+points_rotates[1][1])/2)))
        new_bounding_height.append((points_rotates[1][1]-points_rotates[0][1]))
        new_bounding_width.append((points_rotates[1][0]-points_rotates[0][0]))
        

    
    #cv2.imwrite(path_save+'angle_'+str(rotation)+'_'+file_name+"_not_Noise.jpg", image_incline)
        
    


    return image_incline, new_bounding_box_x, new_bounding_box_y, new_bounding_box_x_center, new_bounding_box_y_center, new_bounding_height, new_bounding_width, rotation, new_bounding_box_x_paper, new_bounding_box_y_paper


def simulate_local_exposure(image, regions=3, mode="mixed", intensity=1.5, zone_size_ratio=0.2):
    """
    zone_size_ratio: float entre 0 et 1, proportion maximale de l'image utilisée comme taille de zone
    """
    img = image.astype(np.float32)
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    min_radius = int(min(h, w) * zone_size_ratio * 0.5)
    max_radius = int(min(h, w) * zone_size_ratio)
    
    for i in range(regions):
        center_x = np.random.randint(w)
        center_y = np.random.randint(h)
        radius = np.random.randint(min_radius, max_radius)
        cv2.circle(mask, (center_x, center_y), radius, i + 1, -1)

    result = img.copy()
    
    for i in range(1, regions + 1):
        region_mask = (mask == i)
        apply_type = mode if mode != "mixed" else np.random.choice(["over", "under"])
        
        for c in range(3):
            if apply_type == "over":
                result[..., c][region_mask] *= intensity
            elif apply_type == "under":
                result[..., c][region_mask] *= (1 / intensity)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def dessiner_grille(image, taille_case=50, couleur=(150, 150, 150), epaisseur=1, alpha=0.5):
    """Ajoute une grille sur une image."""
    h, w, _ = image.shape


    # Étape 1: créer une copie de l'image
    overlay = image.copy()

    

    
    for x in range(0, w, taille_case):
        cv2.line(overlay, (x, 0), (x, h), couleur, epaisseur)
    for y in range(0, h, taille_case):
        cv2.line(overlay, (0, y), (w, y), couleur, epaisseur)

    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    image[:] = blended
    return image


def noise_image(image, seed, bounding_boxes_x, bounding_boxes_y,bounding_boxes_x_paper,bounding_boxes_y_paper, path_save, file_name):
    # Obtenir les dimensions de l'image
    filtre_rouge = np.array([1.0, 0.2, 0.2], dtype=np.float32)  # R, G, B
    filtre_bleu = np.array([0.2, 0.2, 1.0], dtype=np.float32)  # R, G, B
    filtre_vert = np.array([0.2, 1.0, 0.2], dtype=np.float32)  # R, G, B

    np.random.seed(seed)
    filtre_random = np.random.rand(3).astype(np.float32)

    np.random.seed(seed)
    modif_rd = np.random.randint(1,19)
    np.random.seed(seed)
    angle_rd = np.random.randint(0,180)
    np.random.seed(seed)
    vert_rd = np.random.randint(0,20)
    np.random.seed(seed)
    horiz_rd = np.random.randint(0,20)
    np.random.seed(seed)
    region_exposure_rd = np.random.randint(1,5)
    np.random.seed(seed)
    all_filtre = [filtre_rouge, filtre_bleu, filtre_vert, filtre_random]
    filtre = np.random.randint(0, len(all_filtre), 1)[0]
    filtre = all_filtre[filtre]

    # On appliquer des transformations à l'image avec la bibliothèque ECG-Image-Kit
    image_incline, new_bounding_box_x, new_bounding_box_y, new_bounding_box_x_center, new_bounding_box_y_center, new_bounding_height, new_bounding_width, rotation, new_bounding_box_x_paper, new_bounding_box_y_paper = rotate_image(seed,image, bounding_boxes_x, bounding_boxes_y, bounding_boxes_x_paper, bounding_boxes_y_paper)
    
    image_incline = dessiner_grille(np.array(image_incline), taille_case=3, couleur=(150, 150, 150), epaisseur=1, alpha=0.5)
    image_incline = (image_incline*filtre).astype(np.uint8)
    image_incline = simulate_local_exposure(image_incline, regions=region_exposure_rd, mode="mixed", intensity=modif_rd/10, zone_size_ratio=0.2)
    
    _ = get_creased(image_incline,output_file=path_save+file_name+"_Noise_Rotation_"+str(rotation)+"_Seed_"+str(seed)+".png",ifWrinkles=True, modification_file = str(modif_rd)+".jpg", ifCreases=True, crease_angle=angle_rd,num_creases_vertically=vert_rd,num_creases_horizontally=horiz_rd )

    return new_bounding_box_x, new_bounding_box_y, new_bounding_box_x_center, new_bounding_box_y_center, new_bounding_height, new_bounding_width, rotation, modif_rd, angle_rd, vert_rd, horiz_rd,  new_bounding_box_x_paper, new_bounding_box_y_paper




# Chemins des fichiers



# === FONCTION PRINCIPALE POUR TRAITER UN ECG ===
def process_ecg(i, TYPE):
    ### Pour chaque ECG de la base PTB-XL on va construire plusieurs images Clean et bruité
    record_data = df.iloc[i]
    seed = 10 # On pose la seed pour le bruit ajouté à l'image
    seed = record_data["ecg_id"] * seed
    ecg_id = record_data["ecg_id"]
    patient_id = record_data["patient_id"]
    file_name = record_data["filename_hr"].split("/")[-1]

    # Lecture ECG
    ecg = array_to_dict(wfdb.rdsamp(ptb_path + record_data["filename_hr"])[0])
    ### Création de l'image ECG (ecg: dictionnaire avec les signaux des 12 dérivations, path_save: chemin où l'image sera sauvegardée, file_name: nom du fichier, type: type d'image (1 4x4 ou 2 6x2))
    Write_PDF(ecg, path_save + file_name + ".png", TYPE, lead_IIc=ecg["II"])
    ### Lecture de l'image ECG créée
    img = cv2.imread(path_save + file_name + ".png")

    # On définit les coordonnées des bords du papier ECG
    bounding_boxes_x_paper = (24, 500 + 316, 24, 500 + 316)
    bounding_boxes_y_paper = (98, 98, 500 + 52, 500 + 52)

    # Extraction des pistes ECG de l'image avec notre version d'ECGtizer 
    dic_tracks, bounding_boxes_x, bounding_boxes_y, bounding_boxes_x_center, bounding_boxes_y_center, bounding_boxes_width, bounding_boxes_height, possible_overlapping = tracks_extraction(img, "", 300, "multi", NOISE=False, DEBUG=False)

    # On convertit les coordonnées des boîtes englobantes en tableaux numpy pour faciliter la manipulation
    bounding_boxes_x = np.array(bounding_boxes_x)
    bounding_boxes_y = np.array(bounding_boxes_y)
    bounding_boxes_x_center = np.array(bounding_boxes_x_center)
    bounding_boxes_y_center = np.array(bounding_boxes_y_center)
    bounding_boxes_width = np.array(bounding_boxes_width)
    bounding_boxes_height = np.array(bounding_boxes_height)

  




    # Stockage des résultats
    data = []

    # Image originale
    data.append([
        ecg_id, patient_id, seed, 0, 0, 0, 0, bounding_boxes_x, bounding_boxes_y, 
        bounding_boxes_x_center, bounding_boxes_y_center, 
        bounding_boxes_x_paper, bounding_boxes_y_paper,
        bounding_boxes_width, bounding_boxes_height, 
        possible_overlapping, file_name + ".png"
    ])

    # On va appliquer 3 transformations à l'image originale : inclinaison, rotation
    for j in range(3):
        # On récupère l'image originale et les coordonnées des boîtes englobantes
        image_incline, new_bounding_box_x, new_bounding_box_y, new_bounding_box_x_center, new_bounding_box_y_center, new_bounding_height, new_bounding_width, rotation, new_bounding_box_x_paper, new_bounding_box_y_paper = rotate_image(seed, img, bounding_boxes_x, bounding_boxes_y, bounding_boxes_x_paper, bounding_boxes_y_paper)
        
        new_bounding_box_x = np.array(new_bounding_box_x)
        new_bounding_box_y = np.array(new_bounding_box_y)
        new_bounding_box_x_center = np.array(new_bounding_box_x_center)
        new_bounding_box_y_center = np.array(new_bounding_box_y_center)
        new_bounding_box_height = np.array(new_bounding_height)
        new_bounding_box_width = np.array(new_bounding_width)


        # On sauvegarde l'image inclinée
        path_rotation = f"{file_name}_Rotation_{rotation}_Seed_{seed}.png"
        cv2.imwrite(path_save + path_rotation, image_incline)
        
        data.append([
            ecg_id, patient_id, seed, rotation, 0, 0, 0, new_bounding_box_x, new_bounding_box_y, 
            new_bounding_box_x_center, new_bounding_box_y_center,
            new_bounding_box_x_paper, new_bounding_box_y_paper,
            new_bounding_width, new_bounding_height,
            possible_overlapping, path_rotation
        ])
        seed += 1

    # On va incliner et bruiter l'image originale
    for j in range(3):
        # On récupère l'image originale et les coordonnées des boîtes englobantes
        new_bounding_box_x, new_bounding_box_y, new_bounding_box_x_center, new_bounding_box_y_center, new_bounding_height, new_bounding_width, rotation, modif_rd, angle_rd, vert_rd, horiz_rd, new_bounding_box_x_paper, new_bounding_box_y_paper = noise_image(img, seed, bounding_boxes_x, bounding_boxes_y, bounding_boxes_x_paper, bounding_boxes_y_paper, path_save, file_name)
        path_noise = f"{file_name}_Noise_Rotation_{rotation}_Seed_{seed}.png"

        new_bounding_box_x = np.array(new_bounding_box_x)
        new_bounding_box_y = np.array(new_bounding_box_y)
        new_bounding_box_x_center = np.array(new_bounding_box_x_center)
        new_bounding_box_y_center = np.array(new_bounding_box_y_center)
        new_bounding_box_height = np.array(new_bounding_height)
        new_bounding_box_width = np.array(new_bounding_width)


        data.append([
            ecg_id, patient_id, seed, rotation, modif_rd, horiz_rd, vert_rd, new_bounding_box_x, new_bounding_box_y,
            new_bounding_box_x_center, new_bounding_box_y_center,   
            new_bounding_box_x_paper, new_bounding_box_y_paper,
            new_bounding_width, new_bounding_height, 
            possible_overlapping, path_noise
        ])
        seed += 1

    return data

def main(path_data = "ptb_xl/raw/physionet.org/files/ptb-xl/1.0.3/" , path_save = "ptb_xl/noised_images/temp/6x2/", 
        path_save_xml = "../../../../../data/ecg/db_projects/physionet/ptb_xl/noised_images/temp/", TYPE):
    # TYPE = 1 # 4x4 TYPE = 2 # 6x2

    ### Path des données PTB-XL
    ptb_path = path_data
    ### Path où vont être stockées les images
    path_save = path_save
    ### Path où va être stocké le fichier CSV avec les annotations
    path_save_xml = path_save_xml
    ### Fichier d'annotation de la base PTB-XL
    df = pd.read_csv(ptb_path + "ptbxl_database.csv")

    all_data = []
    for i in tqdm(range(len(df))):
        result = process_ecg(i, TYPE)
        all_data.extend(result)

    # === CRÉATION DU DATAFRAME FINAL ===
    columns = ["ecg_id", "patient_id", "seed", "rotation_angle", "texture", "horizontal_light_band", "vertical_light_band",
                "bounding_boxes_x", "bounding_boxes_y", 
            "bounding_boxes_x_center", "bounding_boxes_y_center",
            "bounding_boxes_x_paper", "bounding_boxes_y_paper", 
            "bounding_boxes_width", "bounding_boxes_height", 
            "possible_overlapping", "path"]



    df_all_data = pd.DataFrame(all_data, columns=columns)
    if TYPE == 1:
        df_all_data.to_csv(path_save_xml + "ptbxl_database_4x4_test.csv", index=False)
    elif TYPE == 2:
        df_all_data.to_csv(path_save_xml + "ptbxl_database_6x2_test.csv", index=False)

if __name__ == "__main__":
    # Exécutez la fonction principale avec les chemins appropriés
    main(path_data="ptb_xl/raw/physionet.org/files/ptb-xl/1.0.3/", 
         path_save="ptb_xl/noised_images/temp/6x2/", 
         path_save_xml="../../../../../data/ecg/db_projects/physionet/ptb_xl/noised_images/temp/", TYPE=2)