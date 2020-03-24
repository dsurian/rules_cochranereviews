# Extracting relevant information from Cochrane reviews
This repository contains an updated version of the implementation of rules for automatically extracting relevant information from Cochrane reviews. 

Original implementation: https://github.com/Rabia-Bashir/rules_data_ext/  
Credit to: Rabia Bashir.

### Environment
---
The code was built and tested on:
* python 2.7.16 (Anaconda)
* sklearn: 0.20.3
* numpy: 1.16.5
* OS X El Capitan v10.11.6

### Note
---
Please note that the newer version of some libraries from scikit-learn used in this code might have different default values for the parameters, or, different available parameters, which could give different results.<br />
<br />
### Usage
---
There are 3 python scripts:
1. crawler.py<br />
2. extractor.py<br />
3. classifiers.py<br />

You can create your own Results folder, but the folder should in the same directory with the python scripts and has the same structure, i.e.:
<br />
Main folder<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- crawler.py<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- extractor.py <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- classifier.py <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- Datasets <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- DOI.csv <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- Results <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- HTML_SystematicReviews <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- Your_folder <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- Results <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- HTML_SystematicReviews <br />
<br />

**crawler.py**<br />
The downloaded HTML files are included in HTML_SystematicReviews/ folder.<br />
Re-running the script will download HTML files to your own folder. To run:<br />
&nbsp;&nbsp;&nbsp;&nbsp; ```python crawler.py``` <br /><br />
You will be presented a menu:<br />
&nbsp;&nbsp;&nbsp;&nbsp;```> Enter your folder name:```
<br />
You need to enter your folder name, i.e., **Your_folder**
<br /><br />
This code will read a list of DOI in 'Datasets/DOI.csv' and download the reviews with .pub2 (original version) and .pub3 (updated version) from Cochrane library. The downloaded HTML files are saved to HTML_SystematicReviews folder in Your_Folder (see above).
<br />
Specifically, the crawler will download:<br />
- ```http://cochranelibrary.com/cdsr/doi/{}/full```<br />
- ```http://cochranelibrary.com/cdsr/doi/{}/references```<br />
- ```http://cochranelibrary.com/cdsr/doi/{}/information```<br />
where {} is the DOI
<br />
<br />

**extractor.py**<br />
This code will extract relevant information from the HTML files:
- Search date<br />
&nbsp;&nbsp;&nbsp;&nbsp;Abstract > Search methods, in the HTML file downloaded from ```http://cochranelibrary.com/cdsr/doi/{}/full```, where {} is the DOI<br />
- Number of trials, number of participants in each trial<br />
&nbsp;&nbsp;&nbsp;&nbsp;Characteristics of studies > Characteristics of included studies, in the HTML file downloaded from ```http://cochranelibrary.com/cdsr/doi/{}/references```, where {} is the DOI<br />
- Conclusion<br />
&nbsp;&nbsp;&nbsp;&nbsp;What's New and History, in the HTML file downloaded from ```http://cochranelibrary.com/cdsr/doi/{}/information```<br />
<br />
To run:<br />
&nbsp;&nbsp;&nbsp;&nbsp; ```python extractor.py``` <br /><br />
You will be presented a menu:<br />
&nbsp;&nbsp;&nbsp;&nbsp; ```> Enter your folder name:``` 
<br />
The code will read the HTML files in HTML_SystematicReviews folder in Your_Folder and produce 'extracted_info.txt' in Results folder also in Your_Folder.<br />









* **classifiers.py**
**Running the code**
Type   python classifiers.py    on the console, or run classifiers.py from IDE. A menu will appear:
```
[1] Run default setting and load previous trained classifiers
[2] Run default setting and retrained the classifiers
[3] Enter your own results folder
```
**[1] Default setting and load previous trained classifiers**
This choice will:
- Read 'extracted_info.txt' in 'Results/' folder, and translate to features in 'Results/features.txt'
- Split into 80% for training set (not used), and 20% as test set.
- Load previous trained classifiers in 'Results/cpickle/' folder and reproduce the results on the 20% test set

**[2] Default setting and retrained the classifiers**
This choice will:
- Read 'extracted_info.txt' in 'Results/' folder, and translate to features in 'Results/features.txt'
- Split into 80% for training set (not used), and 20% as test set.
- Retrained classifiers using the 80% training set, and reproduce the results on the 20% test set.

**[3] Using your own results folder**
You can create your own folder to separate the results, however, this folder should be in the same directory with the classifier.py and has the same structure, i.e.:

Main folder
  |- crawler.py
  |- extractor.py
  |- classifier.py
  |- Datasets
  |    |- DOI.csv
  |- Results
  |- HTML_SystematicReviews
  |- **Your_folder**
  |     |- **Results**
  |     |- **HTML_SystematicReviews**


The code contains 3 classifiers: logistic regression, decision tree, and random forest. The classifiers are trained using 80% of dataset and tested using the rest 20%.







# Reference
---
1. *A rule-based approach for automatically extracting data from systematic reviews and their updates to model the risk of conclusion change*. Rabia Bashir, Adam G. Dunn, Didi Surian. Research Synthesis Methods (2020, under review):
