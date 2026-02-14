# Multilayer Perceptron for SDT Classification 
This repository contains a Java implementation of a Multilayer Perceptron (MLP) neural network for solving a supervised discrete classification problem (SDT).

The implementation supports multi-class classification using a fully connected neural network with three hidden layers and configurable activation functions. The model is trained using mini-batch gradient descent and Mean Squared Error (MSE) loss.

The project uses separate training and testing datasets (`train.txt` and `test.txt`) in TXT format and includes functionality for training, evaluation and prediction.



## Project Documentation

A detailed explanation of the MLP implementation, training procedure and SDT classification problem is included in:  
**`Report.pdf`**

The first page of the report lists the **SDT classes** used in the project.



## Dataset Files

- **`train.txt`** – dataset used for training the MLP  
- **`test.txt`** – dataset used for evaluating the MLP  

Both files must follow the format: `x1,x2,label` (no header).



## Features

- Training the network with mini-batch gradient descent  
- Predicting labels for new samples  
- Evaluating accuracy on the test set  
