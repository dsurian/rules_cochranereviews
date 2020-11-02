# Extracting relevant information from Cochrane reviews
This repository contains an updated version of the implementation of rules for automatically extracting relevant information from Cochrane reviews. 

Initial implementation by Rabia Bashir: https://github.com/Rabia-Bashir/rules_data_ext/  

### Environment
---
The code was built and tested on:
* python 2.7.16 (Anaconda)
* scikit-learn: 0.20.3
* SciPy: 1.2.1
* NumPy: 1.16.5
* macOS Catalina 10.15.6

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
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- cpickle <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- **Your_folder** <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- Results <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- HTML_SystematicReviews <br /><br />

**crawler.py**<br />
Re-running the script will download HTML files to your own folder. To run:<br />
&nbsp;&nbsp;&nbsp;&nbsp; ```python crawler.py``` <br />
You will be presented a menu:<br />
&nbsp;&nbsp;&nbsp;&nbsp;```> Enter your folder name:```<br />

You need to enter your folder name, i.e., Your_folder<br />

This code will read a list of DOI in 'Datasets/DOI.csv' and download the reviews with .pub2 (original version) and .pub3 (updated version) from Cochrane library. The downloaded HTML files are saved to HTML_SystematicReviews folder in Your_Folder (see above).<br />

Specifically, the crawler will download:<br />
- ```http://cochranelibrary.com/cdsr/doi/{}/full```<br />
- ```http://cochranelibrary.com/cdsr/doi/{}/references```<br />
- ```http://cochranelibrary.com/cdsr/doi/{}/information```<br />
where {} is the DOI.<br />


**extractor.py**<br />
This code will extract relevant information from the HTML files:
- Search date<br />
&nbsp;&nbsp;&nbsp;&nbsp;Abstract > Search methods, in the HTML file downloaded from ```http://cochranelibrary.com/cdsr/doi/{}/full```, where {} is the DOI<br />
- Number of trials, number of participants in each trial<br />
&nbsp;&nbsp;&nbsp;&nbsp;Characteristics of studies > Characteristics of included studies, in the HTML file downloaded from ```http://cochranelibrary.com/cdsr/doi/{}/references```, where {} is the DOI<br />
- Conclusion<br />
&nbsp;&nbsp;&nbsp;&nbsp;What's New and History, in the HTML file downloaded from ```http://cochranelibrary.com/cdsr/doi/{}/information```<br />

To run:<br />
&nbsp;&nbsp;&nbsp;&nbsp; ```python extractor.py``` <br /><br />
You will be presented a menu:<br />
&nbsp;&nbsp;&nbsp;&nbsp; ```> Enter your folder name:``` <br />

The code will read the HTML files in HTML_SystematicReviews folder in Your_Folder and produce 'extracted_info.txt' in Results folder also in Your_Folder. Alternatively, the 'extracted_info.txt' is also provided in 'Results/' folder.<br />

**classifiers.py**<br />
Type   python classifiers.py    on the console, or run classifiers.py from IDE. A menu will appear:
```
[1] Load previous trained classifiers
[2] Train the classifiers on your dataset
```
**[1] Load previous trained classifiers**<br />
This choice will:<br />
- Read features in 'Results/features.txt'<br />
- Split into 80% for training set (not used), and 20% as test set<br />
- Load previous trained classifiers in 'Results/cpickle/' folder and produce the results on the 20% test set<br />
- Reproduce the reported results<br />

**[2] Train the classifiers on your dataset**<br />
This choice will:<br />
- You will need to enter Your_folder name<br />
- Read 'extracted_info.txt' in 'Your_folder/Results/' folder<br />
- Split into 80% for training set, and 20% as test set.<br />
- Train classifiers using the 80% training set and test on the 20% test set.<br /><br />


The code contains 3 classifiers: logistic regression, decision tree, and random forest. All classifiers were trained using GridSearch to find the best combination of paramaters. Specifically, the tested combinations of parameters for each trained classifiers were:<br />

**Logistic regression**<br />
```
    parameters = {'penalty': ['l1', 'l2'],'class_weight': ['balanced'],'solver': ['liblinear'], ‘C’: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], ’n_jobs': [-1],'random_state':[42]}
```
**Decision tree**<br />
```
    parameters = {"criterion": ["entropy", "gini"],'class_weight': ['balanced'],'max_features': ['auto', 'sqrt', 'log2'],'max_depth': [2, 3, 4],'random_state':[42]}
```
**Random forest**<br />

```
    parameters={'n_estimators': range(5,105,5),'criterion':['entropy','gini'],'class_weight':['balanced'],'max_features':['auto', 'sqrt', 'log2'],'max_depth': [2, 3, 4],'random_state':[42]
```

The combinations of parameters can be found in classifiers.py (run_gridsearchcv_RFClassifier(), run_gridsearchcv_DTClassifier(), run_gridsearchcv_LogisticRegression())





# Reference
---
1. *A rule-based approach for automatically extracting data from systematic reviews and their updates to model the risk of conclusion change*. Rabia Bashir, Adam G. Dunn, Didi Surian. Research Synthesis Methods (2020, under review).
