# ECGtizer

## ECGtizer presentation

ECGtizer is a tool for digitizing ECGs PDF. We have also work to be able to complete the ECG extracted. Indeed the extracted leads are often represented only on 2.5sec or 5sec whearas the record last 10sec. We have developped a model to complete the leads.

## How ECGtizer  works

ECGtizer create a object from a pdf file:

### 1/ First Step: Extraction
```
from ecgtizer import ECGtizer
ecg_extracted = ECGtizer (path, DPI = 500,  verbose = False, DEBUG = False)
``` 
Path : Location of the PDF file\
DPI  : Dot per inch i.e. resolution of the image\
verbose : Indicates the different stages of digitisation and their duration\
DEBUG : Plot image to show where error can occure\

From this object you can have access to the extracted lead with:
```
ecg_extracted.extracted_lead
``` 

### 2/ Second Step: Plot

You can plot the extracted leads with the function *plot*:
```
ecg_extracted.plot(lead = "",  begin = 0, end = 'inf', save = False)
``` 
lead (optional) : The lead you want to plot, if you do not indicate any lead the function will plot all the leads\
begin (optional) : The position on the lead you want to start with\
end (optional) : The position on the lead you want to end with\
save (optinal) : The location to save the plot, if you do not indicate any location the plot will not be saved\
![This is a alt text.](/plot/Plot_all_leads.png)

### 3/ Third Step: Save
You can save the extracted leads into an xml format with the function *save_xml*:
```
extracted_lead.save_xml(save)
``` 
save: The location to save the xml

### 4/ Fourth Step: Completion
You can complete the *missing* information of each leads with the function *completion*:

```
extracted_lead.completion()
``` 
❗❗❗️ The  Completed  leads will replace the extracted leads you can  plot and save the completed leads like you do with the extracted leads ❗❗❗️ 

## Other modules you can use

With the packages analysis you will be able to compare the extracted/completed leads with true leads from a xml files. For that you can use the function *analyse* from the package **analyses**. This function allow to compute a dictionnary with the results of the Pearson correlation / MSE / DTW for each leads between the real ECG and the extracted ones.

```
from analyses import analyse, BlandAltman, scatter_plot, read_xml, alignement, overlap_plot
dic_res = analyse(path_true, path_ex)
``` 
path_true : Location of the xml with the true leads\
path_ex : Location of the xml with the extracted leads

In this package you will find functions to plot the differences between the real leads and the extracted one. You have *overlap_plot*, *scatter_plot* and *BlandAltman* functions:
```
from analyses import overlap_plot
overlap_plot(file1, file2, lead = '', save = False)
```
file1 : Location of the xml with the true leads\
file2 : Location of the xml with the extracted leads\
lead (optional): The lead you want to plot, if you do not indicate any lead all the leads will be plot\
save (optinal) : Location to save the plot\
![This is a alt text.](/plot/Plot_all_leads_overlap.png)

```
from analyses import scatter_plot
scatter_plot(file1, file2, lead = '', save = False)
```
file1 : Location of the xml with the true leads\
file2 : Location of the xml with the extracted leads\
lead (optional): The lead you want to plot, if you do not indicate any lead all the leads will be plot\
save (optinal) : Location to save the plot\
![This is a alt text.](/plot/Plot_all_leads_scatter.png)

```
from analyses import analyse, BlandAltman
BlandAltman(file1, file2, lead = '', save = False)
```
file1 : Location of the xml with the true leads\
file2 : Location of the xml with the extracted leads\
lead (optional): The lead you want to plot, if you do not indicate any lead all the leads will be plot\
save (optinal) : Location to save the plot (❗️If you save the plot it will not be plot)\
![This is a alt text.](/plot/Plot_all_leads_ba.png)

