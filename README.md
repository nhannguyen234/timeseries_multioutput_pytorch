We built this code for a simple problem from a dataset collected from industrial data.

## Problem

The timeseries dataset contains 1 input and 5 output signal. The idea is to build a model which can fit the output data (5 output in this case) if we have only the input. 

![image](https://user-images.githubusercontent.com/33461503/142211454-736a842d-452c-43c1-b76a-831f4535bc4b.png)

## Model

The model was built in Pytorch with some layers of GRU. The model will be improved and updated.

## Results

The best fit can reach to 0.002 Normalized MSE (Normalized to the max absolute value).
![image](https://user-images.githubusercontent.com/33461503/142212067-59546770-1b96-467e-8911-ceb4edaf18b9.png)


## Prerequisites    
Python 3.6+\n
Pytorch 1.19\n
Numpy 1.1.1
