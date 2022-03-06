# Perceptron algorithm in Python
This repo contains code for perceptron algorithm with fixed increment (also called single layer perceptron) implemented in Python. Single layer perceptron is foundation for artificial neural networks. It's my final project for the Pattern recognition university course.

I've written two scripts:

- perceptron_simple.py can learn a linear decision boundary between two classes of 2D samples.
- perceptron_iris.py is an upgraded version where we find decision boundary between each pair of classes. Since class 2 and 3 are not linearly separable, we cannot get the 100% right accuracy.


## Perceptron with fixed incerment algorithm
Perceptron is a binary sorting algorihm with which we can classify two classes of samples that are linearly separable from each other. When creating the project, I first learned the basics of the algorithm. I helped myself to the work with the book Pattern Recognition by Nikola Pavešić.

Vector 𝒘 contains coefficients of the decision boundary. We calculate those coefficients with learning.

The process of learning separation boundaries is as follows:
1. First, increase the dimension of all features to (𝑛 + 1) dimensions, and multiply all learning samples from class two by −1.
2. The vector of coefficients 𝒘 is defined as a vector of zeros or random values, it is only important that it has the same dimension as the extended vector of characteristics (𝑛 + 1).
3. We begin to perform the calculation of the dot products of the separation boundary and the individual sample using equation: 𝑑𝑥= 𝒘^𝑻(1)𝑥(1)
4. If e.g. 𝒘^𝑻(1) = (0,0,0)^𝑇 and 𝒙(1) = (0,0,1)^𝑇 we get the result 0. This means that correction for the 𝒘 is needed.
  𝒘(2)=𝒘(1)+ δ[001]=[000]+1[001]=[001]
  δ is learning rate (in our case 1)
5. When multiplying with the following sample, we take into account the new boundary 𝒘(2).
6. When we get to the end of the samples in the first iteration, we start with the recalculations, from the first sample onwards.
7. We end the training once the decision boundary does not change in one iteration (epoch) of learning. This means that all samples are sorted correctly. Learning can also be interrupted after a certain number of epochs if there is a possibility that the samples are not linearly separable.



### Simple perceptron script
The learning set consists of two classes of samples, each with two samples with two features.

Result for 10 iterations and learning rate equals 1:
![Figure_1](https://user-images.githubusercontent.com/62114221/156884925-f5cce8a9-8d94-4a4a-8634-951ab2e9feb7.png)

Example of command to run the script type:
```console
python perceptron_simple.py "sample.txt" 10 0.1
```
### Perceptron algorithm with Iris dataset
The dataset of this script consists of samples of the Iris dataset in which there are three classes (three types of flowers). Each sample consists of four measurements. The total number of samples is 150, for each class 50.

Iris is a very well-known dataset in the field of pattern recognition. It is characterized by the fact that the combinations of classes 1 and 2 and 1 and 3 are linearly separable, while classes 2 and 3 are not.

Example of command to run the second script:
```console
python perceptron_iris.py "iris.data" 100 0.1
```
