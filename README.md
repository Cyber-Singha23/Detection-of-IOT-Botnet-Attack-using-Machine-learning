# Detection-of-IOT-Botnet-Attack-using-Machine-learning
Machine Learning Based IOT Botnet Detection is a tool to classify network traffic as being botnet affected or not based on the network traffic flows. It involves various classifiers such as Neural Networks, Random Forest, Decision Tree, Naive Bayes, Logistic Regression, k-Nearest Neighbours.

<b> Objective </b>

This project implements a novel method to detect botnet based network intrusion using various Machine Learning based classifiers. Unlike traditional methods based on packet analysis which are inaccurate and time-consuming, this method is robust and highly accurate. This project involves the following machine learning classifiers:

1. Decision Tree
2. Logistic Regression
3. Random Forest
4. Gaussian Naive Bayes
5. K Nearest Neighbours.

<b> Dataset Used </b>

This project uses the UCI dataset which involves data of 9 IOT devices. Each data folder of the IoT device has benign (normal) traffic data with mirai & gafgyt (botnet) traffic. This project uses the 'Philips B120N10 Baby Monitor' data to train and test the various models.

Download the data from here: https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT and then store it in the datset folder.

<b> Dependencies </b>

This project requires set of the following python modules:

1. pandas
2. numpy
3. seaborn
4. scikit-learn
5. matplotlib

<b> Testing the Model </b>

To test the model, run the script.py
