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
Please note that the newer version of some libraries from scikit-learn used in this code might have different default values for the parameters, or, different available parameters, which could give different results.

### Usage
---
There are 3 python scripts:
1. crawler.py
2. extractor.py
3. classifiers.py

You can create your own Results folder, but the folder should in the same directory with the python scripts and has the same structure, i.e.:

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
  
  
**crawler.py**  
The downloaded HTML files are included in 'HTML_SystematicReviews/' folder.

Re-running the script will download HTML files to your own folder. To run:
python crawler.py
You will be presented a menu:
> Enter your folder name:

You need to enter your folder name, i.e., **Your_folder**

This code will read a list of DOI in 'Datasets/DOI.csv' and download the reviews with .pub2 (original version) and .pub3 (updated version) from Cochrane library. The downloaded HTML files are saved to HTML_SystematicReviews folder.

Specifically, the crawler will download:
- ```http://cochranelibrary.com/cdsr/doi/{}/full```
- ```http://cochranelibrary.com/cdsr/doi/{}/references```
- ```http://cochranelibrary.com/cdsr/doi/{}/information```
where {} is the DOI






***** **extractor.py**
**Default setting**
This code will extract relevant information from the respective HTML files:
- Search date
&nbsp;Abstract > Search methods, in the HTML file downloaded from ```http://cochranelibrary.com/cdsr/doi/{}/full```, where {} is the DOI
- Number of trials, number of participants in each trial
&nbsp;Characteristics of studies > Characteristics of included studies, in the HTML file downloaded from ```http://cochranelibrary.com/cdsr/doi/{}/references```, where {} is the DOI
- Conclusion
&nbsp;What's New and History, in the HTML file downloaded from ```http://cochranelibrary.com/cdsr/doi/{}/information```

The output is 'extracted_info.txt' in 'Results/' folder.

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
