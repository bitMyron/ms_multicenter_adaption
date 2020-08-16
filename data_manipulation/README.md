# Data manipulation module
Module to generate feature vectors to train Deep Learning algorithms based on the Theano and Lasagne modules for Python 2.7. With this module we can create patches of 3D images, as well as work on under- and over-sampling of a given feature set (independently of the size).

## Future
The goal is to implement several tests to generate features to test different CNN/CEN configurations. TODO:

- [x] Generate a vector of 3D features to be trained
- [x] Generate test and train data
- [x] Generate CNN examples
- [x] Generate CEN examples (check my other [repository](https://github.com/marianocabezas/miccai_challenge2016) for the nets)
- [x] Integrate this module as a submodule of the miccai challenge repository
- [ ] Work on information theory metrics to determine the relevance of a sample
- [ ] Work on data creation
- [ ] Apply the previous ideas for under-smapling
- [ ] Work on data augmentation (over-sampling)
