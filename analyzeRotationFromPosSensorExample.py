# -*- coding: utf-8 -*-
"""
analyzeRotationFromPosSensorExample.py

example code to process data from position sensor used in centripetal force
experiment

data is stored as a 2 column tab-delimited text file with 
time in the left column and position in the right
only the values near 0 are valid on the position sensor

@author: Marc Gershow
"""

import numpy as np
from posSensorAnalysis import findFrequencyAtZeroCrossing

mbob = 0.4479 #kg -- measured with balance
r_ref = 0.183 #m -- measured with ruler
g = 9.802 #m/s^2 -- http://units.wikia.com/wiki/Gravity_of_Earth
m_force = 0.69 #kg required to stretch weight to reference position -- values stamped on weights used without checking

t,x = np.loadtxt('timeAndPos.txt', unpack=True)

w = findFrequencyAtZeroCrossing(t,x,showPlots = True)
fcent = mbob*r_ref*w**2
fweight = m_force * g
print ("centripetal force = {:.3f} N".format(fcent))
print ("hanging mass force = {:.3f} N".format(fweight))
print ("disagreement = = {:.2f} %".format(2*np.abs(fweight-fcent)/(fweight + fcent)*100))