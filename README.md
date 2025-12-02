# CSC-246-Final-Project
Final Project for CSC 246 - Machine Learning

Authors: Isabel, Johnny, and Nate

Instructions to run:
 - generate with "python main.py --grade 5"
 - train with "python main.py --grade 20 --train 5" where 5 is num of epochs. Doesn't take too long right now.
 - add flag --load to load existing model and pick up training where you left off

File organization:
 - main.py: the main script, used to generate climbs and train the model
 - climb_data.py: contains the custom pytorch Dataset class used to store the data
 - climb_mlp.py: the MLP implementation used for validating model performance
 - climb_util.py: various utility functions, including acquiring the dataset, displaying climbs, and one-hot encoding climbs
 - decoder.py: the main implementation of the transformer decoder model, contaning the generate function
 - encoder.py: implementation of an encoder model, unused in the final project
 - train.py: the functions used to train the model
 - transformer_validating.py: testing the performance of the model using the MLP
   
