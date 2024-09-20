import xmltodict as xml
from scipy.stats import pearsonr
from fastdtw import fastdtw
import numpy as np
import pyCompare
import matplotlib.pyplot as plt

def read_lead(lead_str):
    lead = []
    lead_str = lead_str.split(' ')
    for l in lead_str:
        try:
            lead.append(int(float(l)))
        except ValueError:
            lead.append(0)
           
    return(lead)

def read_xml(file):
    matrix = {}
    with open(file) as fd:
        doc = xml.parse(fd.read())
    
    num_lead = len(doc['AnnotatedECG']['component']['series']['component']['sequenceSet']['component'])
    for i in range(1,num_lead):
        name = doc['AnnotatedECG']['component']['series']['component']['sequenceSet']['component'][i]['sequence']['code']['@code'].split('_')[-1]
        scale = float(doc['AnnotatedECG']['component']['series']['component']['sequenceSet']['component'][i]['sequence']['value']['scale']['@value'])
        lead = read_lead(doc['AnnotatedECG']['component']['series']['component']['sequenceSet']['component'][i]['sequence']['value']['digits'])
        print(scale)
        matrix[name] = np.array(lead) * scale
    return(matrix)


def alignement(lead1, lead2):

    
    a = 0
    if len(lead1) > len(lead2):
        lead1, lead2 = [lead2,lead1]
    if len(lead2) > 5000:
        x = [i for i in range(len(lead2))]
        y = lead2
        new_x = [i for i in np.arange(0,len(lead2),len(lead2)/5000)]
        lead2 = np.interp(new_x,x,y)
    
    score = pearsonr (lead1, lead2[a : a + len(lead1)])[0]
    CONTINUE = True
    
    if len(lead2) != len(lead1):
        list_pos = []
        for a in range(0, len(lead2)-len(lead1)):
            list_pos.append(pearsonr (lead1, lead2[a : a + len(lead1)])[0])
        pos = np.argmax(list_pos)
        lead1 = lead1
        lead2 = lead2[pos:pos+len(lead1)]
    else:
        list_pos_a = []
        list_pos_b = []
        for a in range(0,50):
            list_pos_a.append(pearsonr (lead1[a:], lead2[ : len(lead2) - a])[0])
        for b in range(0,50):
            list_pos_b.append(pearsonr (lead1[: len(lead1) - b], lead2[b : ])[0])
        pos_a = np.argmax(list_pos_a)
        pos_b = np.argmax(list_pos_b)
        
        if pos_a > pos_b:
            lead1 = lead1[pos_a:]
            lead2 = lead2[ : len(lead2) - pos_a]
        else:
            lead2 = lead2[pos_b:]
            lead1 = lead1[ : len(lead1) - pos_b]
    return(lead1, lead2)


def analyse(file1, file2):
    if type(file1) == str:
        file1 = read_xml(file1)
    if type(file2) == str:
        file2 = read_xml(file2)
    
    res_matrix_cor = {}
    res_matrix_mse = {}
    res_matrix_dtw = {}
    dic = {}
    
    for l in file1:
        if l != 'ref':
            l1, l2 = alignement(file1[l], file2[l])
            res_matrix_cor[l] = pearsonr(l1, l2)[0]
            res_matrix_mse[l] = np.mean(np.sqrt(((l1/1000) - (l2/1000))**2))
            res_matrix_dtw[l] = fastdtw((l1/1000), (l2/1000))[0]
        
    dic['corr'] = res_matrix_cor
    dic['mse'] = res_matrix_mse
    dic['dtw'] = res_matrix_dtw
    return(dic)



def BlandAltman(file1, file2, lead = '', save = False):
    if type(file1) == str:
        file1 = read_xml(file1)
    if type(file2) == str:
        file2 = read_xml(file2)
    
    
    
    if lead == '':
        for l in file1:
            if l != 'ref':
                l1, l2 = alignement(file1[l], file2[l])
                if save == False:
                    pyCompare.blandAltman(l1, l2, title = 'Bland-Altman Plot for lead ' + l, pointColour = '#440154', meanColour = '#5ec962', loaColour = '#fde725' )
                else:
                    pyCompare.blandAltman(l1, l2, title = 'Bland-Altman Plot for lead ' + l, savePath = save + l + '.png', pointColour = '#440154', meanColour = '#5ec962', loaColour = '#fde725')
    else:
        
        l1, l2 = alignement(file1[lead], file2[lead])
        if save == False:
            pyCompare.blandAltman(l1, l2, title = 'Bland-Altman Plot for lead ' + lead, pointColour = '#440154', meanColour = '#5ec962', loaColour = '#fde725')
        else:
            pyCompare.blandAltman(l1, l2, title = 'Bland-Altman Plot for lead ' + lead, savePath = save + lead + '.png', pointColour = '#440154', meanColour = '#5ec962', loaColour = '#fde725')
 
        

        
def compute_slope(l1, l2):
    r, _ = pearsonr(l1, l2)
    std_Y = np.std(l1)
    std_X = np.std(l2)
    slope = r * (std_Y / std_X)
    return(slope)
    
def scatter_plot(file1, file2, lead = '', save = False):
    if type(file1) == str:
        file1 = read_xml(file1)
    if type(file2) == str:
        file2 = read_xml(file2)
    
    if lead == '':
        for l in file1:
            if l != 'ref':
                plt.figure(figsize = (10,7))
                l1, l2 = alignement(file1[l], file2[l])
                plt.scatter(l1, l2, color = '#440154',  s = 3)
                    
                # Calculer la droite de régression linéaire
                coefficients = np.polyfit(l1, l2, 1)
                pente = coefficients[0]
                intercept = coefficients[1]
                
                x_values = np.linspace(l1.min(), l1.max(), 100)
                
                y_values = intercept + pente * x_values
                
                plt.plot(x_values, y_values, color='#fde725')
                
                plt.text(l1.min(), l2.max(),f'r : {pente:.2f}', fontsize=12, weight = 'bold')
                plt.title ('Scatter Plot for lead ' + l)
                plt.xlabel('Extracted lead')
                plt.ylabel('True lead')
                if save != False:
                    plt.savefig(save + '_' + l + '.png')
                plt.show()
    else:
        plt.figure(figsize = (10,7))
        l1, l2 = alignement(file1[lead], file2[lead])
        
        # Calculer la droite de régression linéaire
        coefficients = np.polyfit(l1, l2, 1)
        pente = coefficients[0]
        intercept = coefficients[1]

        x_values = np.linspace(l1.min(), l1.max(), 100)

        y_values = intercept + pente * x_values

        plt.plot(x_values, y_values, color='#fde725')
        plt.text(l1.min(), l2.max(),f'r : {pente:.2f}', fontsize=12, weight = 'bold')
        plt.title ('Scatter Plot for lead ' + lead)
        plt.scatter(l1, l2, color = '#440154', s = 3)
        plt.xlabel('Extracted lead')
        plt.ylabel('True lead')
        if save != False:
            plt.savefig(save + '_' + lead + '.png')
        plt.show()
    
    
def overlap_plot(file1, file2, lead = '', save = False):
    if type(file1) == str:
        file1 = read_xml(file1)
    if type(file2) == str:
        file2 = read_xml(file2)
    
    if lead == '':
        for l in file1:
            if l != 'ref':
                plt.figure(figsize = (10,7))
                l1, l2 = alignement(file1[l], file2[l])
                plt.plot(l1, color = 'r', label = 'Extracted ECG')
                plt.plot(l2, color = 'g', label = 'True ECG')

                plt.title ('Overlap plot for lead ' + l)
                plt.xlabel('True lead')
                plt.ylabel('Extracted lead')
                plt.legend()
                if save != False:
                    plt.savefig(save + '_' + l + '.png')
                plt.show()
    else:

        l1, l2 = alignement(file1[lead], file2[lead])

        plt.figure(figsize = (10,7))
        plt.plot(l1, color = 'r', label = 'Extracted ECG')
        plt.plot(l2, color = 'g', label = 'True ECG')

        plt.title ('Overlap plot for lead ' + lead)
        plt.xlabel('True lead')
        plt.ylabel('Extracted lead')
        plt.legend()
        if save != False:
            plt.savefig(save + '_' + l + '.png')
        plt.show()
    
    
    
    
    