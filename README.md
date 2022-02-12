# Tumor-Classification

## Problem Statement:
Brain Scan Tumor Classification using a Convolutional Neural Network

Given an image, detect if it has a brain tumor associated with it.



## Motivation
The Brain Scan Tumor Classification is a machine learning project that classifies 2D brain scan images as tumorous or not.

Model evaluation will be based on performance of the classification by the final model after hyperparameter tuning.

## Dataset information
The Brain tumor dataset contains 2 folders 
yes: this folder contains all the images having brain tumor
no: this folder contain all the images that shows no brain tumor

## Classes: 
The label for the data can be
yes-for tumorous
no- for non-tumorous
The dataset needs to be processed the size of the images vary and dimensions can be expanded using numpy array


## Steps:

#### Clone the repository 

#### To install all required packages to run this project type following in your terminal
<pip install -r requirements.txt>

#### Go to api folder
 < cd api >

#### Run main file using the following command
 < python main.py >

#### Check your results using postman by entering the request url:
1) To check health:
        http://localhost:8000/ping

2) To give prediction
        http://localhost:8000/predict



