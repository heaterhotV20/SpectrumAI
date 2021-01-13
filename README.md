# Spectrum AI

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Spectrum AI uses computer vision technique to classify the different types of radio frequency signals emitted in the environment.


### Tech

The RF receiver used is a hackrf software defined radio that has the ability to capture from 50MHz to 6GHz frequency range. In this experiment, up to L band frequencies with known emissions are captured. The frequency sweep mode at 100Hz rate is constant for a fixed start and end frequency range.

reGenData processes the sweep file captured by the HackRF device nad does some cleaning such as filling up the empty values in each frequency bin.

trimSpectrum identifies start and end frequencies in the spectrogram image and trims them into smaller images of fixed height (25 time intervals) and width 20 Mhz.

train.py uses the trimmed images for training with a 0.2 validation split and saves it in model file

test.py tests the trained model on test images

An overall accuracy of about 90% is achieved with a 11 unique emitter classes.



### Installation

Spectrum AI requires Python 3 libraries to run.

