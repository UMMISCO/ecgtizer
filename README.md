# ECGtizer

## ECGtizer presentation

ECGtizer is a tool for digitizing ECGs PDF. We have also work to be able to complete the ECG extracted. Indeed the extracted leads are often represented only on 2.5sec or 5sec whearas the record last 10sec. We have developped a model to complete the leads.

## Installation

Ensure you have the required dependencies installed by creating a Conda environment using the provided `ecgtizer_env.yml` file.

```bash
conda env create -f ecgtizer_env.yml
conda activate ecgtizer
```

## How ECGtizer  works

ECGtizer create a object from a pdf file:

## How to use ECGtizer with bash

```
python ECGtyzer_main.py "data/PTB-XL/PDF/00121_hr.pdf" 500 "fragmented" --verbose "data/PTB-XL/Digitized/test.xml"
```

## How to use ECGtizer with Jupyter Notebook

### 1/ First Step: Extraction
```
from ecgtizer import ECGtizer
ecg_extracted = ECGtizer (path, DPI = 500,extraction_method="fragmented",  verbose = False, DEBUG = False)
``` 
path : Location of the PDF file\
extraction_method: Method of extraction ("full", "lazy", "fragemented")\
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
completion(optional, bool): If you want to plot the complete lead,
save (optinal) : The location to save the plot, if you do not indicate any location the plot will not be saved\


### 3/ Third Step: Save
You can save the extracted leads into an xml format with the function *save_xml*:
```
extracted_lead.save_xml(save)
``` 
save: The location to save the xml

### 4/ Fourth Step: Completion
You can complete the *missing* information of each leads with the function *completion*:

```
ecg_extracted.completion(model_path, device)
``` 
model_path (str): location of the model to use for the completion in .pth format\
device (torch.device): device to use for the completion
❗❗❗️ The  Completed  leads will replace the extracted leads you can  plot and save the completed leads like you do with the extracted leads ❗❗❗️ 


