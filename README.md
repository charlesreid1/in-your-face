# in-your-face

Neural networks for facial recognition using Keras and the Labeled Faces in the Wild (LFW) Database.

Table of Contents:
* [List of software used](#software)
* [LFW dataset](#lfw)
	* [Training and testing data splits](#training_testing)
* [Running scripts](#running)
* [Useful links](#links)
* [Labeled Faces in the Wild](#lfw)
* [Keras](#keras)



<a name="software"></a>
## List of Software Used

Python:
* python 3
* jupyter notebook
* numpy
* scipy
* pandas
* pillow (Python image library) 

Machine learning/neural networks:
* scikit-learn
* tensorflow
* keras

Data loading packages:
* [fuel](https://github.com/mila-udem/fuel) - provides data wrappers for machine learning
* [kerosene](https://github.com/dribnet/kerosene) - provides a way to give fuel data sets 
	various splits, version numbers, etc.
* [charlesreid1/lfw_fuel](https://github.com/charlesreid1/lfw_fuel) - fork of [dribnet/lfw_fuel](https://github.com/dribnet/lfw_fuel),
	provides a way to load LFW data into Fuel format

```
#!/bin/bash

# py for sci
pip3 install jupyter numpy scipy pandas pillow

# learn you some machines
pip3 install sklearn tensorflow keras

# get you some data for your learn
cd /tmp
git clone https://github.com/mila-udem/fuel 
cd fuel && python3 setup.py build && python3 setup.py install && cd ../

git clone https://github.com/dribnet/kerosene 
cd kerosene && python3 setup.py build && python3 setup.py install && cd ../

git clone https://github.com/charlesreid1/lfw_fuel 
cd lfw_fuel && python3 setup.py build && python3 setup.py install && cd ../
```

<a name="lfw"></a>
## LFW Data Set

See [LFW README.txt](http://vis-www.cs.umass.edu/lfw/README.txt) for detailed information about the data set.

See [lfw-names.txt](http://vis-www.cs.umass.edu/lfw/lfw-names.txt) for a list of all individuals in the dataset and the corresponding number of images of that person.

The LFW data set provided on the LFW website is provided in multiple formats:
* Original data set - contains unaligned faces (this can be difficult for facial recognition systems)
* Funneled data sets - funneled data sets re-orient the faces to be consistent
* Subset of images: only people with A names (smaller, good for exploring how to import/process)
* Subset of images: only George W Bush (most images of single person, also good for exploring importing/processing)

<a name="training_testing"></a>
### Training and Testing Data Splits

The training data and the test data are contained in two different text files:
* pairsDevTest.txt
* pairsDevTrain.txt

Each file contains two sections, corresponding to positive matches and negative matches.

The first section (positive matches) contains a single name and two numbers:

```
Alan_Greenspan	1	5
```

This indicates that the neural network is to be fed the 1st and 5th photograph
of Alan Greenspan, and told that the target answer is "Yes," i.e., that these two 
photographs are of the same person. 

The second section (negative matches) contains two names and two numbers:

```
Rod_Stewart	3	Se_Hyuk_Joo	1
```

This indicates that the neural network is to be fed the 3rd photograph of Rod Stewart
and the 1st photograh of Se Hyuk Joo and told that the target answer is "No," i.e.,
that these two photographs are of different people.

See the [lfw_fuel](https://github.com/charlesreid1/lfw_fuel) repository for a nice 
Python wrapper 

From [LFW README.txt](http://vis-www.cs.umass.edu/lfw/README.txt):

```
2a. Image Restricted Configuration
----------------

In the first formulation, the training information is restricted to
the image pairs given in the pairs.txt file.  No information about the
actual names of the people in the image pairs should be used.  This is
meant to address the issue of transitivity.

In other words, if one matched pair consists of the 10th and 12th
images of George_W_Bush, and another pair consists of the 42nd and
50th images of George_W_Bush, then under this formulation it would not
be allowable to use the fact that both pairs consist of images of
George_W_Bush in order to form new pairs such as the 10th and 42nd
images of George_W_Bush.

To ensure this holds, one should only use the name information to
identify the image, but not provide the name information to the actual
algorithm.  For this reason, we refer to this formulation as the Image
Restricted Configuration.  Under this formulation, only the pairs.txt
file is needed.
```


<a name="running"></a>
## Running Scripts

The scripts are all contained in Jupyter notebooks. Spin up a Jupyter notebook server using the command:

```
jupyter notebook
```

<a name="links"></a>
## Useful Links

<a name="lfw"></a>
### Labeled Faces in the Wild (LFW) 

[Link to the LFW Face Database](http://vis-www.cs.umass.edu/lfw/)

[Link to resources related to the LFW Face Database](http://vis-www.cs.umass.edu/lfw/#resources)

[LFW Book Chapter (Academic)](https://people.cs.umass.edu/~elm/papers/LFW_survey.pdf)

<a name="keras"></a>
### Keras

[charlesreid1/lfw_fuel run-lfw.py example](https://github.com/charlesreid1/lfw_fuel/blob/master/example/run-lfw.py) 
contains a simple Keras convolutional neural network that illustrates how to connect the LFW dataset (in fuel format)
to the Keras neural network, and how to assemble a simple convolutional neural network architecture.


