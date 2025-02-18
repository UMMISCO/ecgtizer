import os
import matplotlib.pyplot as plt
import xml.etree.cElementTree as ET
from xml.etree import ElementTree
from os.path import isfile,join, isdir, exists
from os import listdir, makedirs
import xmltodict as xml
import numpy as np


def plot_function(lead_all,lead = '', b = 0, e = 'inf', c = None, save = False, transparent = False):
    """
    Main function to write the XML from the numpy extracted ECG
    
    Parameters
    ----------
    lead_all     : dictionary, dictionary of all the extracted leads
    lead         : str, lead name to plot
    b            : int, start of the plot
    e            : int , end of the plot
    save         : str, the location to save the plot
    
    Returns
    -------
    void
    """
    # If it is a multilead or not
    if len(lead_all) > 1:
        # If it is Kardia, 3x4 or 6x2 format
        # if it is 3x4
        if len(lead_all) == 13:
            FORMAT = [3,4]
            dic_pos = {
                'I'   : [0,0],
                'II'  : [1,0],
                'III' : [2,0],
                'AVR' : [0,1],
                'AVL' : [1,1],
                'AVF' : [2,1],
                'V1'  : [0,2],
                'V2'  : [1,2],
                'V3'  : [2,2],
                'V4'  : [0,3],
                'V5'  : [1,3],
                'V6'  : [2,3]
                
            }
        # If it is 6x2
        elif len(lead_all) == 12:
            FORMAT = [6,2]
            dic_pos = {
                'I'   : [0,0],
                'II'  : [1,0],
                'III' : [2,0],
                'AVR' : [3,0],
                'AVL' : [4,0],
                'AVF' : [5,0],
                'V1'  : [0,1],
                'V2'  : [1,1],
                'V3'  : [2,1],
                'V4'  : [3,1],
                'V5'  : [4,1],
                'V6'  : [5,1]
                
            }
        # If it is Kardia multilead
        elif len(lead_all) == 7:
            FORMAT = [6,1]
            dic_pos = {
                'I'   : [0],
                'II'  : [1],
                'III' : [2],
                'AVR' : [3],
                'AVL' : [4],
                'AVF' : [5],
                
            }
        if lead != '':
            plt.title(lead)
            plt.plot(lead_all[lead], c = c)
            plt.xlabel('Time (1/500)sec')
            plt.ylabel('Amplitude µV')
        else:
            fig = plt.figure(figsize=(20,14))
            axs = fig.subplots(FORMAT[0],FORMAT[1])
            for i in lead_all:
                if i != 'ref' and i != 'IIc':
                    if len(dic_pos[i]) == 2 :
                        axs[dic_pos[i][0],dic_pos[i][1]].plot(lead_all[i], c = c)
                        axs[dic_pos[i][0],dic_pos[i][1]].set_title('lead '+i)
                        axs[dic_pos[i][0],dic_pos[i][1]].set_xlabel('Time (1/500)sec')
                        axs[dic_pos[i][0],dic_pos[i][1]].set_ylabel('Amplitude µV')
                    else:
                        axs[dic_pos[i][0]].plot(lead_all[i], c = c)
                        axs[dic_pos[i][0]].set_title('lead '+i)
                        axs[dic_pos[i][0]].set_xlabel('Time (1/500)sec')
                        axs[dic_pos[i][0]].set_ylabel('Amplitude µV')
                
            fig.tight_layout()
            if save != False:
                name = save
                a = 1
                while os.path.exists(name):
                    path = ''
                    name_temp = name.split('/')

                    for i in range(len(name_temp)-1):
                        path += name_temp[i] + '/'
                    name_temp = name_temp[-1]
                    name_temp = name_temp.split('.')
                    if name_temp[0][-1] == ')':
                        name_temp1 = name_temp[0].split('(')
                        name = path + name_temp1[0] + '(' + str(a) +').'+name_temp[1]
                    else:
                        name = path + name_temp[0]+'('+str(a)+').'+name_temp[1]
                    a+=1
                print(name)
                plt.savefig(name, transparent=transparent)
            plt.show()
    else:
        if e == 'inf':
            plt.plot(lead_all['all'][b:], c = c)
            
        else:
            plt.plot(lead_all['all'][b:e], c = c)
        if e == 'inf':
            e = len(lead_all['all'])
        plt.title('ECG from ' + str(int(b/500)) +'sec to '+str(int(e/500))+'sec')
        plt.xlabel('Time (1/500)sec')
        plt.ylabel('Amplitude µV')
        if save != False:
            name = save
            a = 1
            while os.path.exists(name):
                path = ''
                name_temp = name.split('/')

                for i in range(len(name_temp)-1):
                    path += name_temp[i] + '/'
                name_temp = name_temp[-1]
                name_temp = name_temp.split('.')
                if name_temp[0][-1] == ')':
                    name_temp1 = name_temp[0].split('(')
                    name = path + name_temp1[0] + '(' + str(a) +').'+name_temp[1]
                else:
                    name = path + name_temp[0]+'('+str(a)+').'+name_temp[1]
                a+=1
            print(name)
            plt.savefig(name, transparent=transparent)
        plt.show()

def plot_overlay(lead,image, piqueh, piquev):
    plt.imshow(image, cmap = 'gray')
    for i in range(len(piqueh)):
        median = np.median(lead[i])
        aj = piqueh[i] - median
        beg = [np.nan for i in range(piquev)]
        plt.plot(np.concatenate((beg,lead[i] + aj)), c = 'r')

        

def write_lead_root(code_lead, codeSystem, code_lead_name, scale_y):
    root = ET.Element("sequence")
    branch = ET.subElement(root,'code')
    subbranch = ET.subElement(branch,'code')
    subbranch.set ('code',code_lead)
    subbranch = ET.subElement(branch,'codeSystem')
    subbranch.set('codeSystem',codeSystem)
    subbranch = ET.subElement(branch,'codeSystemName')
    subbranch.set('codeSystemName',code_lead_name)
    branch = ET.subElement(root, 'value')
    branch.set('@xsi:type', 'SLIST_PQ')
    subranch = ET.subElement(branch, 'origin')
    subranch.set('value', '0')
    subranch.set('unit', 'uV')
    subranch = ET.subElement(branch, 'scale')
    subranch.set('value', str(scale_y/10))
    subranch.set('unit', 'uV')
    ET.SubElement(branch, "digits").text = lead
    return (root)

def write_element(code_lead, codeSystem, actcode, scale_y):
    root = ET.Element("sequence")
    branch = ET.subElement(root,'code')
    subbranch = ET.subElement(branch,'code')
    subbranch.set ('code','TIME_RELATIVE')
    subbranch = ET.subElement(branch,'codeSystem')
    subbranch.set('codeSystem',codeSystem)
    subbranch = ET.subElement(branch,'codeSystemName')
    subbranch.set('codeSystemName',actcode)
    branch = ET.subElement(root, 'value')
    branch.set('@xsi:type', 'SLIST_PQ')
    subranch = ET.subElement(branch, 'head')
    subranch.set('value', '0.000')
    subranch.set('unit', 's')
    subranch = ET.subElement(branch, 'increment')
    subranch.set('value', str((scale_x/10)/500))
    subranch.set('unit', 's')
    return (root)



def writexml (path_out, time_low, time_high,date_version, num_version, id_root, code, rhythm, actcode, codeSystem, displayName, manufacturerModelName, manufacturerOrganization, code_lead, code_lead_name, scale_x, scale_y, lead, type_pdf, 
Low_freq, High_freq, BPM, PR, QRS, QT, QTc, P, R, T, moyRR, QTcB, QTcF, Rythme, ECG, Age, sex, other):
    root = ET.Element("AnnotatedECG")
    root .set("xsi:schemaLocation","urn:hl7-org:v3 /HL7/aECG/2003-12/schema/PORT_MT020001.xsd")
    root .set("xmlns","urn:hl7-org:v3")
    root .set("xmlns:voc","urn:hl7-org:v3/voc")
    root .set("xmlns:xsi","http://www.w3.org/2001/XMLSchema-instance")


    # Will be change but I do not know how
    # ID branch
    branch = ET.SubElement(root, 'id')
    branch .set('root',"999c4713-7232-4b38-82fb-98644dafceef")


    # CODE Branch
    branch = ET.SubElement(root, 'code')
    branch .set('code','9300')
    branch .set('codeSysteme','2.16.840.1.113883.6.12')


    # VERSION Branch
    branch = ET.SubElement(root, 'version')
    branch .set('date',date_version)
    branch .set('version',num_version)




    # EFECTIVE TIME Branch

    branch = ET.SubElement(root, 'effectiveTime')
    subbranch = ET.SubElement(branch, 'low')
    subbranch.set('value',time_high)
    
    subbranch = ET.SubElement(branch, 'high')
    subbranch.set('value',time_low)

    
    ########################## COMPONENTOF Branch ###########################################
    branch = ET.SubElement(root, 'componentOf')
    subbranch = ET.SubElement(branch, 'timepointEvent')
    subbranch .set('code', 'None')

    
    subsubbranch = ET.SubElement(subbranch, 'componentOf')
    subsubsubbranch = ET.SubElement(subsubbranch, 'subjectAssignment')
    
        ##################### SUBJECT SubBranch ########################################
    subsubsubsubbranch = ET.SubElement(subsubsubbranch, 'subject')
    subsubsubsubsubbranch = ET.SubElement(subsubsubsubbranch,'trialSubject')
    subsubsubsubsubsubbranch = ET.SubElement(subsubsubsubsubbranch,'id')	
    subsubsubsubsubsubbranch .set('extension','None')

        ##################### End SUBJECT SubBranch ###################################
    
        ##################### COMPONENTOF SubBranch ###################################
    subsubsubsubbranch = ET.SubElement(subsubsubbranch, 'componentOf')
    subsubsubsubsubbranch = ET.SubElement(subsubsubsubbranch, 'clinicalTrial')
    subsubsubsubsubbranch .set('id','None')
    subsubsubsubsubbranch .set('title','None')
    
            ############# COMPONENTOF SubSubBranch ################################
    subsubsubsubsubsubbranch = ET.SubElement(subsubsubsubsubbranch, "componentOf")
    subsubsubsubsubsubsubbranch = ET.SubElement(subsubsubsubsubsubbranch, "clinicalTrialProtocol")
    subsubsubsubsubsubsubbranch .set('name','None')
    subsubsubsubsubsubsubbranch .set('id','None')

                ############# AUTHOR SubSubBranch ################################
    subsubsubsubsubsubbranch = ET.SubElement(subsubsubsubsubbranch, "author")
    subsubsubsubsubsubsubbranch = ET.SubElement(subsubsubsubsubsubbranch, "clinicalTrialSponsor")
    subsubsubsubsubsubsubsubbranch = ET.SubElement(subsubsubsubsubsubsubbranch, "sponsorOrganization")
    
    subsubsubsubsubsubsubsubbranch .set('id','None')
    subsubsubsubsubsubsubsubbranch .set('name','None')

                ############ LOCATION SubSubBranch ###############################
    subsubsubsubsubsubbranch = ET.SubElement(subsubsubsubsubbranch, "location")
    subsubsubsubsubsubsubbranch = ET.SubElement(subsubsubsubsubsubbranch, "trialSite")
    
    subsubsubsubsubsubsubsubbranch = ET.SubElement(subsubsubsubsubsubsubbranch, "id")
    subsubsubsubsubsubsubsubbranch .set('extension','0')

    
    subsubsubsubsubsubsubsubbranch = ET.SubElement(subsubsubsubsubsubsubbranch, "location")
    subsubsubsubsubsubsubsubbranch .set('name','None')
    

                ############ responsibleParty SubSubBranch ###############################
    subsubsubsubsubsubbranch = ET.SubElement(subsubsubsubsubbranch, "responsibleParty")
    subsubsubsubsubsubsubbranch = ET.SubElement(subsubsubsubsubsubbranch, "trialInvestigator")
    
    subsubsubsubsubsubsubsubbranch .set('id','None')

    
    subsubsubsubsubsubsubsubbranch = ET.SubElement(subsubsubsubsubsubsubbranch, "investigatorPerson")	
    subsubsubsubsubsubsubsubbranch .set('name','None')

    ########################## END COMPONENTOF Branch #######################################	
    
    ########################## COMPONENTOF Branch ###########################################
    branch = ET.SubElement(root, 'component')
    subbranch = ET.SubElement(branch, 'series')

                ############ Type of ECG #########################################
    subsubbranch = ET.SubElement(subbranch, 'ECG_type')
    subsubbranch.set('type',type_pdf)
    
                ############ ID SubSubBranch #####################################
    subsubbranch = ET.SubElement(subbranch, 'id')
    subsubbranch.set('root',id_root)
    
                ############ CODE SubSubBranch #####################################
    subsubbranch = ET.SubElement(subbranch, 'code')
    subsubbranch.set('code',rhythm)
    subsubbranch.set('codeSystemName',actcode)
    subsubbranch.set('codeSystem',codeSystem)
    subsubbranch.set('displayName', displayName)


                    ############ EFFECTIVETIME SubSubBranch #####################################
    subsubbranch = ET.SubElement(subbranch, 'effectiveTime')
    
    subsubsubbranch = ET.SubElement(subsubbranch, 'low')
    subsubsubbranch.set('value',time_low)

    
    subsubsubbranch = ET.SubElement(subsubbranch, 'high')
    subsubsubbranch.set('value',time_high)


                    ############ AUTHOR SubSubBranch #####################################
    subsubbranch = ET.SubElement(subbranch, 'author')
    subsubsubbranch = ET.SubElement(subsubbranch, 'seriesAuthor')
    subsubsubsubbranch = ET.SubElement(subsubsubbranch , 'manufacturedSeriesDevice')
    subsubsubsubsubbranch = ET.SubElement(subsubsubsubbranch , 'id')
    
    subsubsubsubsubbranch.set('extension','0')
    subsubsubsubbranch.set('manufacturerModelName',manufacturerModelName)
    
    
    subsubsubsubbranch = ET.SubElement(subsubsubbranch, 'manufacturerOrganization')
    subsubsubsubbranch.set('name',manufacturerOrganization)
    
                    ############ COMPONENT SubSubBranch #####################################
    subsubbranch = ET.SubElement(subbranch, 'component')
    subsubsubbranch = ET.SubElement(subsubbranch, 'sequenceSet')
    
    component = ET.SubElement(subsubsubbranch, 'component')
    
    subbranch7 = ET.SubElement(component,'sequence')
    subbranch8 = ET.SubElement(subbranch7,'code')
    
    subbranch8.set ('code','TIME_ABSOLUTE')
    
    subbranch8.set('codeSystem',codeSystem)
    
    subbranch8.set('codeSystemName',actcode)
    
    subbranch8.set('displayName', 'absolute Time')
    
    
    subbranch8 = ET.SubElement(subbranch7, 'value')
    subbranch8.set('xsi:type', 'SLIST_PQ')
    
    subranch9 = ET.SubElement(subbranch8, 'head')
    subranch9.set('value',time_low)
    
    subranch9 = ET.SubElement(subbranch8, 'increment')
    subranch9.set('value', str((int(scale_x)/10)/500))
    subranch9.set('unit', 's')
    
    for l in lead:
        componentbranch = ET.SubElement(subsubsubbranch, 'component')
        subbranch7 = ET.SubElement(componentbranch, 'sequence')
        
        subbranch8 = ET.SubElement(subbranch7, 'code')
        subbranch8.set ('code',code_lead+"_"+l)
        
        subbranch8.set('codeSystemName',code_lead_name)

        subbranch8.set('codeSystem',codeSystem)

    
        
        
        subbranch8 = ET.SubElement(subbranch7, 'value')
        subbranch8.set('xsi:type', 'SLIST_PQ')
        
        subbranch9 = ET.SubElement(subbranch8, 'origin')
        subbranch9.set('value', '0')
        subbranch9.set('unit', 'uV')
        
        subbranch9 = ET.SubElement(subbranch8, 'scale')
        subbranch9.set('value', str(1))
        subbranch9.set('unit', 'uV')
    

        ET.SubElement(subbranch8, "digits").text = lead[l]
    
    
            ############ DERIVATION SubSubBranch #####################################
    subsubbranch = ET.SubElement(subbranch, 'derivation')
    subsubsubbranch = ET.SubElement(subsubbranch, 'derivedSeries')

                        ############ ID SubSubSubBranch ##################################
    subsubsubsubbranch = ET.SubElement(subsubsubbranch, 'id')
    subsubsubsubbranch.set('root',id_root)
    
                        ############ CODE SubSubSubBranch ##################################
    subsubsubsubbranch = ET.SubElement(subsubsubbranch, 'code')
    subsubsubsubbranch.set('code',code)
    subsubsubsubbranch.set('codeSystem',codeSystem)
    subsubsubsubbranch.set('codeSystemName',actcode)
    subsubsubsubbranch.set('displayName',displayName)
    
    
                    ############ EFFECTIVETIME SubSubSubBranch ##################################
    subsubsubsubbranch = ET.SubElement(subsubsubbranch, 'effectiveTime')
    
    subsubsubsubsubbranch = ET.SubElement(subsubsubsubbranch, 'low')
    subsubsubsubsubbranch.set('value',time_high)
    
    subsubsubsubsubbranch = ET.SubElement(subsubsubsubbranch, 'high')
    subsubsubsubsubbranch.set('value',time_high)

    
                    ############ COMPONENT SubSubSubBranch ##################################
    subsubsubsubbranch = ET.SubElement(subsubsubbranch, 'component')
    subsubsubsubsubbranch = ET.SubElement(subsubsubsubbranch, 'sequenceSet')
    componentbranch = ET.SubElement(subsubsubsubsubbranch, 'component')
    
    subbranch7 = ET.SubElement(componentbranch,'sequence')
    subbranch8 = ET.SubElement(subbranch7,'code')
    
    subbranch8.set ('code','TIME_RELATIVE')
    
    subbranch8.set('codeSystem',codeSystem)
    
    subbranch8.set('codeSystemName',actcode)
    
    subbranch8.set('displayName', 'Relative Time')
    
    
    subbranch8 = ET.SubElement(subbranch7, 'value')
    subbranch8.set('xsi:type', 'SLIST_PQ')
    
    subranch9 = ET.SubElement(subbranch8, 'head')
    subranch9.set('value', '0.000')
    subranch9.set('unit', 's')
    
    subranch9 = ET.SubElement(subbranch8, 'increment')
    subranch9.set('value', str((int(scale_x)/10)/500))
    subranch9.set('unit', 's')
    


    for l in lead:
        componentbranch = ET.SubElement(subsubsubsubsubbranch, 'component')
        subbranch7 = ET.SubElement(componentbranch, 'sequence')
        
        subbranch8 = ET.SubElement(subbranch7, 'code')
        subbranch8.set ('code',code_lead+"_"+l)
        
        subbranch8.set('codeSystemName',code_lead_name)

        subbranch8.set('codeSystem',codeSystem)

        
        
        
        subbranch8 = ET.SubElement(subbranch7, 'value')
        subbranch8.set('xsi:type', 'SLIST_PQ')
        
        subbranch9 = ET.SubElement(subbranch8, 'origin')
        subbranch9.set('value', '0')
        subbranch9.set('unit', 'uV')
        
        subbranch9 = ET.SubElement(subbranch8, 'scale')
        subbranch9.set('value', str(int(scale_y)/10))
        subbranch9.set('unit', 'uV')
        

        ET.SubElement(subbranch8, "digits").text = lead[l]



                        ############ SUBJECTOF SubSubSubBranch ##################################
    subsubsubsubbranch = ET.SubElement(subsubsubbranch, 'subjectOf')
    subsubsubsubsubbranch = ET.SubElement(subsubsubsubbranch, 'annotationSet')
    
    componentbranch = ET.SubElement(subsubsubsubsubbranch, 'component')
    annotationbranch = ET.SubElement(componentbranch, 'Annotation')
    annotationbranch.set ('Scalex', str(int(scale_x)))
    annotationbranch.set ('Scaley', str(int(scale_y)))
    annotationbranch.set ('LowFreq', str(Low_freq))
    annotationbranch.set ('highFreq', str(High_freq))
    annotationbranch.set ('BPM', str(BPM))
    annotationbranch.set ('InterPR', str(PR))
    annotationbranch.set ('DurQRS', str(QRS))
    annotationbranch.set ('QT', str(QT))
    annotationbranch.set ('QTc', str(QTc))
    annotationbranch.set ('AxesP',str(P))
    annotationbranch.set ('AxesR',str(R))
    annotationbranch.set ('AxesT',str(T))
    annotationbranch.set ('MoyRR',str(moyRR))
    annotationbranch.set ('QTcB',str(QTcB))
    annotationbranch.set ('QTcF',str(QTcF))
    annotationbranch.set ('Rythme', str(Rythme))
    annotationbranch.set ('ECG', str(ECG))
    annotationbranch.set ('sex',str(sex))
    annotationbranch.set ('age',str(Age))
    annotationbranch.set ('other',str(other))


    tree = ET.ElementTree(root)
    
    test = path_out.split("/")[:-1]
    try:
        
        direction = ""
        for t in test:
            direction += t + "/"

        if not exists(direction):
            makedirs(direction)
        tree.write(path_out)
    except FileNotFoundError:
        tree.write(path_out)


# We transform the np file into a string
def transform_np2txt(List):
    List = List.tolist()
    txt = ''
    for i in range(len(List)-1):
        txt+=str(List[i])+" "
    txt+=str(List[-1])
    return(txt)

# We transform time into a correct format
def conversion_time (day, month, year, hour):
    month_dic = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07','Aug':'08',
                 'Sep':'09','Oct':'10','Nov':'11','Dec':'12', 'unknow' : '00'}
    if hour == 'unknow':
        hour = ['00','00']
    else:
        hour = hour.split(":")
    if day == 'unknow':
        day = '00'
    if year == 'unknow':
        year = '0000'
    convert = ''
    convert = year+month_dic[month]+day
    for h in hour:
        convert+=h
    return(convert)

# Main function
def write_xml (matrix, path_out, TYPE = '', table = '',  num_version = '0.0', date_version  = "27.O6.2022"): 
    """
    Main function to write the XML from the numpy extracted ECG
    
    Parameters
    ----------
    matrix       : dictionary, dictionary of all the extracted leads
    path_out     : str, the location to save the xml
    TYPE         : str, format of the image
    table        : df , dataframe with all the extracted information from the image
    num_version  : str, the software version number
    date_version : str, data of the last version
    
    Returns
    -------
    void
    """
    
    # Define a dictionnary with for each lead name the associated signal in string
    dic_lead_str ={}
    for l in matrix:
        if l!= 'ref':
            dic_lead_str[l] = transform_np2txt(np.array(matrix[l]))

    # We define the time for xml
    time_low = conversion_time('unknow', 'unknow' , 'unknow','unknow')
    type_pdf = TYPE
    time_high = "0"


        
    # we define the id root
    id_root ='d0c6bf1d-566d-46e3-9dff-f0d1727b8d4e'
    
    # We define the information about rhythm and system
    rhythm = 'RHYTHM'
    actcode = 'ActCode'
    codeSystem = '2.16.840.1.113883.5.4'
    displayName = 'Rhythm Waveforms'
    code = 'REPRESENTATIVE_BEAT'

    # We define the engine information 
    manufacturerModelName = 'el250'
    manufacturerOrganization = 'Mortara Instrument, Inc.'
    
    # We define the code for lead
    code_lead = 'MDC_ECG_LEAD' 
    code_lead_name = 'MDC'
    
    # We define the Scale
    scale_y = "1"
    scale_x = "1"
    # We define all annotated information of the PDF #
    Low_freq = table['low_freq']
    High_freq = table['high_freq']
    BPM = table['BPM']
    PR = table['Inter PR (ms)']
    QRS = table['Dur.QRS (ms)']
    QT = table['QT (ms)']
    QTc = table['QTc (ms)']

    P = table['Axe P']
    R = table['Axe R']
    T = table['Axe T']

    RR = table['Moy RR (ms)']
    QTcB = table['QTcB (ms)']
    QTcF = table['QTcF (ms)']

    Rythme = table['Rythme']
    ECG = table['ECG']
    
    Age = table['Age']
    sex = table['sex']
    other = table['other_information']

        
    # We write the xml file
    writexml (path_out,time_low, time_high,date_version, num_version, id_root, 
              code,rhythm, actcode, codeSystem,displayName,manufacturerModelName,
              manufacturerOrganization,code_lead, code_lead_name, scale_x, scale_y, 
              dic_lead_str, type_pdf, Low_freq, High_freq, BPM, PR, QRS, QT, QTc, P, 
              R, T, RR, QTcB, QTcF, Rythme, ECG, Age, sex, other)



