#!/usr/bin/python

import math
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    flattened_predictions = [val for sublist in predictions.tolist() for val in sublist]
    flattened_ages = [val for sublist in ages.tolist() for val in sublist]
    flattened_net_worths = [val for sublist in net_worths.tolist() for val in sublist]

    ### your code goes here
    for prediction, age, net_worth in zip(flattened_predictions, flattened_ages, flattened_net_worths):
        error = math.pow(prediction - net_worth, 2)
        datatuple = (age, net_worth, error)
        cleaned_data.append(datatuple)


    cleaned_data.sort(key=lambda data: data[2])
    cleaned_data = cleaned_data[0:81]
    return cleaned_data

