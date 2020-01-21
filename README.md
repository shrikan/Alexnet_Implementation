Alexnet_Implementation
======================

Implementation of paper ImageNet Classification with Deep Convolutional Neural Networks. </br>
Refer: https://www.cs.toronto.edu/~fritz/absps/imagenet.pdf

------------------------------------------------------------------------------------------------------------------------
**About Alexnet**</br>
AlexNet is the name of a convolutional neural network, designed by Alex Krizhevsky and published with Ilya Sutskever  </br>
and Krizhevsky's PhD advisor Geoffrey Hinton, who was originally resistant to the idea of his student.  </br>

AlexNet competed in the ImageNet Large Scale Visual Recognition Challenge on September 30, 2012.  </br>
The network achieved a top-5 error of 15.3%, more than 10.8 percentage points lower than that of the runner up. </br>
The original paper's primary result was that the depth of the model was essential for its high performance,  </br>
which was computationally expensive, but made feasible due to the utilization of graphics processing units (GPUs) </br>
during training.

* source : WikiPedia
------------------------------------------------------------------------------------------------------------------------

The given dataset itself is divided into training, valiadtion and test dataset.

**Data** </br>
You can obtain dataset from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

**Pre-requisites**
1. Python3</br> 
2. Expects certain libraries to be installed prior to running this job </br> 
      Thus, As a one time activity, run pre_check.py from the scripts folder to handle it.
      
**How to run**
1. Create a folder 'data' parllel to src and dump the data from above link
2. cd to src folder
3. Two ways to run it:
   - If you don't have other test set and just want to run end to end : </br>
      python run.py
   - If you have a separate set of test set and want to run model on that:
      python model_training.py   (to train)
      python model_test.py .     (to test)
