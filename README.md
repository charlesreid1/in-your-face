# in-your-face

Neural networks for facial recognition using Keras and the Labeled Faces in the Wild (LFW) Database.

## List of Software Used

Python:
* Python 3
* Jupyter Notebook
* Numpy
* Scipy
* Pandas
* Pillow (Python image library) 

`pip3 install jupyter numpy scipy pandas pillow`

Machine learning/neural networks:
* Scikit-learn
* Tensorflow
* Keras

`pip3 install sklearn tensorflow keras`

## LFW Data Set

See [LFW README.txt](http://vis-www.cs.umass.edu/lfw/README.txt) for detailed information about the data set.

See [lfw-names.txt](http://vis-www.cs.umass.edu/lfw/lfw-names.txt) for a list of all individuals in the dataset and the corresponding number of images of that person.

The LFW data set provided on the LFW website is provided in multiple formats:
* Original data set - contains unaligned faces (this can be difficult for facial recognition systems)
* Funneled data sets - funneled data sets re-orient the faces to be consistent
* Subset of images: only people with A names (smaller, good for exploring how to import/process)
* Subset of images: only George W Bush (most images of single person, also good for exploring importing/processing)

### Training and Testing Data Splits

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



## Running Scripts

The scripts are all contained in Jupyter notebooks. Spin up a Jupyter notebook server using the command:

```
jupyter notebook
```

## Useful Links

### Labeled Faces in the Wild (LFW) 

[Link to the LFW Face Database](http://vis-www.cs.umass.edu/lfw/)

[Link to resources related to the LFW Face Database](http://vis-www.cs.umass.edu/lfw/#resources)

[LFW Book Chapter (Academic)](https://people.cs.umass.edu/~elm/papers/LFW_survey.pdf)

### Keras

[Github user dribnet: run-lfw.py](https://github.com/dribnet/lfw_fuel/blob/master/example/run-lfw.py)

