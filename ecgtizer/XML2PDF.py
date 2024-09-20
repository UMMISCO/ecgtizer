#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parses an ECG file produced by the Contec ECG90A electrocardiograph
and produces a graph in PDF (vector) or PNG (raster) format.

Required custom modules: ecg_contec.py, which requires ecg_scp.py
Required Python packages: python3-numpy python3-scipy python3-reportlab
"""


from os.path import isfile,join, isdir, exists
from os import listdir, makedirs


import argparse
import math
import os.path
import subprocess
import sys
import warnings
import numpy as np
import xmltodict as xml
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt, iirnotch
from scipy.ndimage import uniform_filter
from reportlab.graphics.shapes import Drawing, Line, PolyLine, String, Group, colors
from reportlab.lib.units import mm, inch
from reportlab.lib.colors import HexColor
from reportlab.graphics import renderPDF, renderPM
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os


def read_lead(lead_str):
    lead = []
    lead_str = lead_str.split(' ')
    for l in lead_str:
        try:
            lead.append(int(float(l)))
        except ValueError:
            lead.append(np.nan)
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
        matrix[name] = np.array(lead) * scale
    return(matrix)


########################### Class Plot ####################################
class ecg_plot():

    # Default unit of measure is mm, suitable for PDF (vector) output.
    DEFAULT_UNIT = mm


    # Graphical elements size (in mm).
    DEFAULT_PAPER_W = 297.0
    DEFAULT_PAPER_H = 210.0
    MARGIN_LEFT = 8.0
    MARGIN_RIGHT = 8.0
    MARGIN_TOP = 5.0
    MARGIN_BOTTOM = 15.0
    FONT_SIZE = 2.7
    FONT_SMALL_SIZE = 3.50
    THICK_LINE = 0.24
    THIN_LINE = 0.00000001

    # Set default rows and cols (in reality we could have either 6/2 or 4/3)
    DEFAULT_ROWS = 6
    DEFAULT_COLS = 2
    DEFAULT_SPEED = 25.0  # Default X-axis scale is 25 mm/s
    DEFAULT_LEADS_TO_PLOT = list(range(0, 12))
    LEAD_LABEL = (u'I', u'II', u'III', u'aVR', u'aVL', u'aVF', u'V1', u'V2', u'V3', u'V4', u'V5', u'V6')

    LINEJOIN_MITER = 0
    LINEJOIN_ROUND = 1
    LINEJOIN_BEVEL = 2

    # The following values can be changed before calling add_lead_plots().
    # X-distance between polot points (in self.unit).
    PLOT_PITCH = 0.2
    # Apply an uniform_filter() if each PLOT_PITCH covers more than MIN sample points.
    UNIFORM_FILTER_MIN_PTS = 4
    # Use scipy.signal.lfilter() instead of scipy.signal.filtfilt() for low-pass filtering.
    USE_LFILTER = False

    # Define a graphic style for pdf
    class line_style():
        def __init__(self, strokeColor=colors.black, strokeWidth=1, strokeLineCap=0, strokeLineJoin=0, strokeMiterLimit=0, strokeDashArray=None, strokeOpacity=None):
            self.strokeColor = strokeColor
            self.strokeWidth = strokeWidth
            self.strokeLineJoin = strokeLineJoin

    # Define a caligraphic style for pdf
    class string_style():
        def __init__(self, fontName='Times-Roman', fontSize=10, fillColor=colors.black, textAnchor='start'):
            self.fontName = fontName
            self.fontSize = fontSize
            self.fillColor = fillColor
            self.textAnchor = textAnchor

############################################ Functions to draw ##############################################################
    def __init__(self, unit=DEFAULT_UNIT, paper_w=DEFAULT_PAPER_W, paper_h=DEFAULT_PAPER_H, cols=DEFAULT_COLS, rows=DEFAULT_ROWS, ampli=None, speed=DEFAULT_SPEED):
        pdfmetrics.registerFont(TTFont('sans-cond', 'fonts/DejaVuSansCondensed.ttf'))
        pdfmetrics.registerFont(TTFont('sans-mono', 'fonts/DejaVuSansMono.ttf'))
        pdfmetrics.registerFont(TTFont('sans-mono-bold', 'fonts/DejaVuSansMono-Bold.ttf'))
        self.unit = unit
        self.paper_w = paper_w
        self.paper_h = paper_h
        self.cols = cols
        self.rows = rows
        self.time0 = 0.0
        self.ampli = ampli
        self.speed = speed
        self.leads_to_plot = self.DEFAULT_LEADS_TO_PLOT
        self.notch = None
        # Calculated sizes.
        #Definition de la zone "record"
        self.graph_w = int((self.paper_w - (self.MARGIN_LEFT + self.MARGIN_RIGHT)) / 10.0) * 10.0
        self.graph_h = int((self.paper_h - (self.MARGIN_TOP + self.MARGIN_BOTTOM) - self.FONT_SIZE * 8) / 10.0) * 10.0
        
        
        self.graph_x = (self.paper_w - self.graph_w) / 2.0
        self.graph_y = self.MARGIN_BOTTOM
        self.time1 = self.time0 + ((self.graph_w / self.cols)  / self.speed)
        if self.ampli is None:
            self.ampli = int((self.graph_h / (self.rows * 1.8)) / 5) * 5.0
        # Calculated styles.
        self.sty_line_thick  = self.line_style(strokeColor=HexColor('#f2c3c3'), strokeWidth=(self.THICK_LINE*self.unit)/10)
        self.sty_line_blue   = self.line_style(strokeColor=colors.blue, strokeWidth=self.THICK_LINE*self.unit)
        self.sty_line_thin   = self.line_style(strokeColor=HexColor('#eecfce'), strokeWidth=self.THIN_LINE*self.unit)
        self.sty_line_plot   = self.line_style(strokeColor=colors.black, strokeWidth=self.THICK_LINE*self.unit, strokeLineJoin=self.LINEJOIN_ROUND)
        self.sty_str_bold    = self.string_style(fontName='sans-mono-bold', fontSize=self.FONT_SIZE*self.unit)
        self.sty_str_regular = self.string_style(fontName='sans-cond', fontSize=self.FONT_SMALL_SIZE*self.unit)
        self.sty_str_blue    = self.string_style(fontName='sans-mono', fontSize=self.FONT_SMALL_SIZE*self.unit, fillColor=colors.blue)
        # Calculate how many sample points there are for each plot pitch.
        self.samples_per_plot_pitch = 1 + int(500 / self.speed * self.PLOT_PITCH)
        # Preapre the drawing.
        self.draw = Drawing(paper_w*self.unit, paper_h*self.unit)


    def axis_tick(self, x, y, s):
        return self.draw_line(x, y-0.5, x, y+3, s)


    def draw_polyline(self, points, s):
        return PolyLine(points, strokeColor=s.strokeColor, strokeWidth=s.strokeWidth, strokeLineJoin=s.strokeLineJoin)


    def draw_line(self, x0, y0, x1, y1, s):
        return Line(x0 * self.unit, y0 * self.unit, x1 * self.unit, y1 * self.unit, strokeColor=s.strokeColor, strokeWidth=s.strokeWidth)


    def draw_text(self, x, y, text, s):
        s1 = String(x * self.unit, y * self.unit, text)
        s1.fontName = s.fontName
        s1.fontSize = s.fontSize
        s1.fillColor = s.fillColor
        s1.textAnchor = s.textAnchor
        return s1


    def ticks_positions(self, x_min, x_max, mm_per_x_unit):
        """ Calculate where to place the ticks over bottom X axis """
        ticks = {}
        axis_len = x_max - x_min
        # Start searching a suitable span from the power of 10 above axis_len.
        if axis_len < 1.0:
            e = 10 ** int(math.log10(axis_len))
        else:
            e = 10 ** (int(math.log10(axis_len)) + 1)
        m = 1
        divs = 0
        while divs < 5:
            span = m * e
            divs = int(axis_len / span)
            if m == 1:
                e = e / 10.0
                m = 5
            elif m == 5:
                m = 2
            else:
                m = 1
        if (x_min % span) == 0:
            first_tick = (int(x_min / span)) * span
        else:
            first_tick = (int(x_min / span) + 1) * span
        # Stop ticks 5mm before the axis end.
        last_tick = x_max - (5.0 / mm_per_x_unit)
        for x_val in np.arange(first_tick, last_tick, span):
            position = (x_val - x_min) * mm_per_x_unit
            ticks[position] = x_val
        return ticks


    def lead_plot_points(self, yp, x_offset, y_offset, width, freq):
        """ Return the point coordinates (in self.unit) for one lead graph """
        # Coordinates are shifted into the page by (x_offset, y_offset).
        plot_points = []
        start = 0.0
        stop = width
        step = self.PLOT_PITCH
        # X-axis points for np.interp()
        xp = np.array(range(0, len(yp)))
        
        
        for x in np.arange(start, stop, step):
            if ( self.time0 + (x / self.speed)) * freq < len(yp):
                sample = ( self.time0 + (x / self.speed)) * freq
                if sample > 0:
                    y = np.interp(sample, xp, yp)
                    if not np.isnan(y) :
                        y = y * 1000 / 1000000.0 * self.ampli
                        px = (x_offset + x) * self.unit
                        py = (y_offset + y) * self.unit
                        plot_points.extend((px, py))
          
        return plot_points


    def iirnotch_filter(self, data, cutoff, fs):
        """ Apply a band-stop filter at the specified cutoff frequency """
        # The quality (-3 dB threshold) is set at cutoff +/- 3 Hz.
        w0 = cutoff / (fs * 0.5)
        quality = cutoff / 6.0
        b, a = iirnotch(w0, quality)
        y = lfilter(b, a, data)
        return y



    def add_graph_paper(self):
        """ Draw graph paper: thick/thin horizontal/vertical lines """
        x0 = self.graph_x
        x1 = self.graph_x + self.graph_w
        y0 = self.graph_y
        y1 = self.graph_y + self.graph_h
        step = 1.0
        for x in np.arange(x0, x1+0.1, step):
            self.draw.add(self.draw_line(x, y0, x, y1, self.sty_line_thin))
        for y in np.arange(y0, y1+0.1, step):
            self.draw.add(self.draw_line(x0, y, x1, y, self.sty_line_thin))
        step = 5.0
        for x in np.arange(x0, x1+0.1, step):
            self.draw.add(self.draw_line(x, y0, x, y1, self.sty_line_thick))
        for y in np.arange(y0, y1+0.1, step):
            self.draw.add(self.draw_line(x0, y, x1, y, self.sty_line_thick))

############################################# End Functions to draw ##############################################################
    
############################################# Functions to draw information ######################################################
    
    ########## Information about confidentiality number, age and sex ##############
    def add_ID_data(self, filename, database, age, sex):
        col_left = (
            '%s' % (filename,),
            'ID: %s' % (database,),
            '%s ans, %s' % (age,sex)
        )
        x = self.graph_x
        y = self.paper_h - self.MARGIN_TOP - self.FONT_SIZE * 1.125
        for d in col_left:
            self.draw.add(self.draw_text(x, y, d, self.sty_str_bold))
            y -= self.FONT_SIZE * 1.125
            
    ########## Information about ecg ###############################################        
    def add_info_data(self, date ):
        """ Print file and case info """
        col_left = (
            '%s' % (date,),
            'Fréq.Vent:',
            'Int PR:',
            'Dur.QRS:',
            'QT/QTc:',
            'Axes P-R-T:',
            'Moy RR:',
            'QTcB:',
            'QTcF:'
        )
        x = (self.graph_x + self.graph_w / 2.0)/2
        y = self.paper_h - self.MARGIN_TOP - self.FONT_SIZE * 1.125 
        for d in col_left:
            self.draw.add(self.draw_text(x, y, d, self.sty_str_bold))
            y -= self.FONT_SIZE * 1.125
            
         
    def add_mid_data(self,frq_vent, int_pr, QRS, QT_QTc, Axes, Moy_RR, QTcB, QTcF):
        """ Print file and case info """
        col_left = (
            '' ,
            '%s bpm' % (frq_vent,),
            '%s ms'  % (int_pr,),
            '%s ms'  % (QRS,),
            '%s ms'  % (QT_QTc,),
            '%s'     % (Axes,),
            '%s ms'  % (Moy_RR,),
            '%s ms'  % (QTcB,),
            '%s ms'  % (QTcF,)
        )
        x = (self.graph_x + self.graph_w / 2.0)/2+20
        y = self.paper_h - self.MARGIN_TOP - self.FONT_SIZE * 1.125 
        for d in col_left:
            self.draw.add(self.draw_text(x, y, d, self.sty_str_bold))
            y -= self.FONT_SIZE * 1.125
           

    ############# Other information ###############
    def add_other_data(self, info1,info2,info3):
        """ Print patient data """
        col_right = (
            '%s' % (info1,),
            '%s' % (info2,),
            '%s' % (info3,),
        )
        x = self.graph_x + self.graph_w / 2.0
        y = self.paper_h - self.MARGIN_TOP - self.FONT_SIZE * 1.125
        for d in col_right:
            self.draw.add(self.draw_text(x, y, d, self.sty_str_bold))
            y -= self.FONT_SIZE * 1.125


    ############ Information on scaling ##########
    def add_plot_info_text(self):
        """ Text above and below the graph paper """
        x = self.graph_x + 1.0
        y = self.MARGIN_BOTTOM - self.FONT_SMALL_SIZE * 1.125
        text = u'Speed: %.2fmm/s %s Leads: %.2fmm/mV' % (self.speed, u' '*6, self.ampli)
        self.draw.add(self.draw_text(x, y, text, self.sty_str_regular))

    ############ Information on the freq, the machine and other information (not finish yet) #########
    def add_plot_filter_text(self, freq):
        """ Filter description below the graph paper """
        x = self.graph_x + (self.graph_w / 2) - 40
        y = self.MARGIN_BOTTOM - self.FONT_SMALL_SIZE * 1.125
        text = u'Sample Rate: %dHz ' % freq
        self.draw.add(self.draw_text(x, y, text, self.sty_str_regular))

    ################ Plot lead ######################################
    def add_lead_plots(self,impulse_pulse, data, freq, type_of_pdf,complet, offset=0 ):
        """ Lead plots, aligned into a grid of ROWS x COLS """
        ticks = self.ticks_positions(self.time0, self.time1, self.speed)
        # Divide the record into sector for each lead
        sector_w = self.graph_w / self.cols
        sector_h = self.graph_h / self.rows
        k = 0
        for c in range(0, self.cols):

            # Add the ticks over the X axis.
            for pos in ticks:
                x = pos + self.graph_x + sector_w * c
                # Echelle bleu en bas
                #self.draw.add(self.axis_tick(x, self.MARGIN_BOTTOM, self.sty_line_blue))
                #self.draw.add(self.draw_text(x+0.5, self.MARGIN_BOTTOM+0.5, '%.1f' % ticks[pos], self.sty_str_blue))
            
            if type_of_pdf == 'type1':
                range_max = self.rows-1
            else:
                range_max = self.rows   
                
                
                         
            for r in range(0, range_max):

                if k >= len(self.leads_to_plot):
                    break
                         
                # Plot le nom des leads
                i = self.leads_to_plot[k]
                label = self.LEAD_LABEL[i]
                x0 = self.FONT_SIZE + self.graph_x + sector_w * c
                y0 = (self.graph_y + self.graph_h) - self.FONT_SIZE - sector_h * r
                self.draw.add(self.draw_text(x0+10, y0, label, self.sty_str_bold))

                
                
                # Filtration des leads
                filt_data = data[i]
                applied_filters = []
                # If many points per pitch, apply an uniform_filter on them.
                if self.samples_per_plot_pitch >= self.UNIFORM_FILTER_MIN_PTS:
                    applied_filters.append(u'uniform_filter(size=%d)' % (self.samples_per_plot_pitch,))
                    filt_data = uniform_filter(filt_data, self.samples_per_plot_pitch)

                y_offset = self.graph_y + self.graph_h - sector_h * (r + 0.5) - offset 
                 
                # Case of type2 PDF#
                if type_of_pdf == 'type2':
                    if i < 6 :
                        p = self.lead_plot_points(impulse_pulse, 10, y_offset, sector_w, freq)
                        if len(p) > 1:
                   	     self.draw.add(self.draw_polyline(p, self.sty_line_plot))
                        x_offset_dep = 17.2
                        x_offset = x_offset_dep
                    else :
                        x_offset = x_offset_dep + sector_w - 15
                        
                # Case of type1 PDF #
                else:
                    if i < 3 :
                    	 # Reference Pulse for tree first Lead #
                        p = self.lead_plot_points(impulse_pulse, 8.3, y_offset, sector_w, freq)
                        if len(p) > 1:
                   	     self.draw.add(self.draw_polyline(p, self.sty_line_plot))
                   	 
                   	 # Reference Pulse for the last One IIc #
                        y_offset_last = self.graph_y + self.graph_h - sector_h * (self.rows-1 + 0.5) - offset
                        p = self.lead_plot_points(impulse_pulse, 8.3, y_offset_last, sector_w, freq)
                        if len(p) > 1:
                            self.draw.add(self.draw_polyline(p, self.sty_line_plot))
                   	 
                        x_offset_dep = 16
                        x_offset = x_offset_dep
                    elif i < 6 :
                        x_offset_sec = x_offset_dep + sector_w -8
                        x_offset = x_offset_sec
                    elif i < 9 : 
                        x_offset_third = x_offset_sec + sector_w - 8
                        x_offset = x_offset_third
                    else:
                        x_offset_four = x_offset_third + sector_w - 8
                        x_offset = x_offset_four
                    

                #print(u'%3s: %s' % (label, '; '.join(applied_filters)))
                p = self.lead_plot_points(filt_data, x_offset, y_offset, sector_w, freq)

                if len(p) > 1:
                    self.draw.add(self.draw_polyline(p, self.sty_line_plot))
                k += 1
                
                
        if type_of_pdf == 'type1':
            if self.samples_per_plot_pitch >= self.UNIFORM_FILTER_MIN_PTS:
                    applied_filters.append(u'uniform_filter(size=%d)' % (self.samples_per_plot_pitch,))
                    filt_data = uniform_filter(complet, self.samples_per_plot_pitch)        
            p = self.lead_plot_points(filt_data, x_offset_dep, y_offset_last, sector_w*4, freq)
            if len(p) > 1:
                self.draw.add(self.draw_polyline(p, self.sty_line_plot))
                
                
def Write_PDF (ecg, path_output, type_of_pdf, lead_IIc = ""):
    initial_dir = os.getcwd()
    path_output = initial_dir + '/' + path_output

    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    impulse_pulse = []
    for i in range(140):
        if i < 20 or i >140-20:
            impulse_pulse.append(0)
        else:
            impulse_pulse.append(1000)
			
    File_name ='test'
    output_units = mm
    ampli = 10
    speed = 25

        
    new_ecg = {}

    if type_of_pdf == 'type2':
        cols = 2
        rows = 6
        leads_to_plot = list(range(0, 12))
        #dic_lead = {'impulse1':0, 'impulse2':1,'impulse1':2, 'impulse2':3,'impulse1':4, 'impulse2':5,'I':6,'II':7,'III':8,'AVR':9,'AVL':10,'AVF':11,'V1':12,'V2':13,'V3':14,'V4':15,'V5':16,'V6':17}
        #for i in range(6):
        #	new_ecg[i] = impulse_pulse
        
    elif type_of_pdf == 'type1':
        leads_to_plot = list(range(0, 12))
        cols = 4
        rows = 4
        #dic_lead = {'impulse1':0, 'impulse2':1, 'impulse3':2, 'I':3,'II':4,'III':5,'AVR':6,'AVL':7,'AVF':8,'V1':9,'V2':10,'V3':11,'V4':12,'V5':13,'V6':14}
        
        #for i in range(3):
        #	new_ecg[i] = impulse_pulse

    # Prepare the Reportlab Drawing object.
    plot = ecg_plot(unit=output_units, cols=cols, rows=rows, ampli=ampli, speed=speed)

    plot.leads_to_plot = leads_to_plot
    plot.add_graph_paper()

    freq = 500

    if type_of_pdf == 'type1':
        for k in ecg.keys():
            if k == "I":
                new_ecg[0] = ecg[k][:1250]
            elif k == "II":
                new_ecg[1] = ecg[k][:1250]
            elif k == "III":
                new_ecg[2] = ecg[k][:1250]
            elif k == "aVR" or k == "AVR":
                new_ecg[3] = ecg[k][1250:2500]
            elif k == "aVL" or k == "AVL":
                new_ecg[4] = ecg[k][1250:2500]
            elif k == "aVF" or k == "AVF":
                new_ecg[5] = ecg[k][1250:2500]
            elif k == "V1":
                new_ecg[6] = ecg[k][2500:3750]
            elif k == "V2":
                new_ecg[7] = ecg[k][2500:3750]
            elif k == "V3":
                new_ecg[8] = ecg[k][2500:3750]
            elif k == "V4":
                new_ecg[9] = ecg[k][3750:5000]
            elif k == "V5":
                new_ecg[10] = ecg[k][3750:5000]
            elif k == "V6":
                new_ecg[11] = ecg[k][3750:5000]

    elif type_of_pdf == 'type2':
        for k in ecg.keys():
            if k == "I":
                new_ecg[0] = ecg[k][:2500]
            elif k == "II":
                new_ecg[1] = ecg[k][:2500]
            elif k == "III":
                new_ecg[2] = ecg[k][:2500]
            elif k == "aVR" or k == "AVR":
                new_ecg[3] = ecg[k][:2500]
            elif k == "aVL" or k == "AVL":
                new_ecg[4] = ecg[k][:2500]
            elif k == "aVF" or k == "AVF":
                new_ecg[5] = ecg[k][:2500]
            elif k == "V1":
                new_ecg[6] = ecg[k][2500:]
            elif k == "V2":
                new_ecg[7] = ecg[k][2500:]
            elif k == "V3":
                new_ecg[8] = ecg[k][2500:]
            elif k == "V4":
                new_ecg[9] = ecg[k][2500:]
            elif k == "V5":
                new_ecg[10] = ecg[k][2500:]
            elif k == "V6":
                new_ecg[11] = ecg[k][2500:]

    
    plot.add_lead_plots(impulse_pulse, new_ecg, freq, type_of_pdf, lead_IIc)
            


    pdf_title = 'ECG %s %dx%d t0=%.1fsec' % ('ECG', rows, cols, 0.0)
    renderPDF.drawToFile(plot.draw, path_output, pdf_title)
    #print(u'INFO: Saved file "%s"' % (path_output.split("/")[-1]))    
    os.chdir(initial_dir)       
                
                
def xml_to_pdf(path_input, path_output,  type_of_pdf = 'type1'):
    #print(os.listdir())
    ecg = read_xml(path_input)
    Write_PDF (ecg, path_output, type_of_pdf)
    
                
                
                
                
                
                
                
                
                
                
