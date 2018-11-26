# -*- coding: utf-8 -*-
"""
posSensorAnalysis.py
Created on Sun Nov 25 19:58:04 2018

code to find frequency of rotation at reference displacement
for use with position sensor extension to centripetal force lab

key function: findFrequencyAtZeroCrossing

@author: Marc Gershow
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def smooth(x, sigma):
    """
    smooths data by doing a gaussian blur
    parameters
    ----------
        x : np.array
        sigma : standard deviation of blur
    returns
    -------
        np.array 
            same size as x, blurred
            
    """
    w = signal.gaussian(np.ceil(6*sigma), sigma)
    w = w / sum(w)
    return signal.convolve(x,w,mode='same')

def findFrequencyAtZeroCrossing(t, x, smoothTau = 0.01, crossingRange = 0.002, showPlots = False):
    """
    finds the angular frequency of rotation (omega) when mass crosses reference point
    parameters
    ----------
        t : np.array
            position sensor measurement times
        x : np.array
            position sensory measurements -- only values near 0 are meaningful
        smoothTau : float
            how much to smooth position measurements to avoid spurious peaks
            value expressed in seconds
        crossingRange : float
            distance (in mm) from reference point 
            only peak locations within this distance of 0 will be used
        showPlots : boolean
            if true, shows a few plots highlighting key analysis steps
    returns
    -------
        omega : float
            angular frequency of rotation at center location
    """
    dt = np.median(np.diff(t))
    xs = smooth(x,smoothTau/dt)
    
    deltar = -xs #distance from r_ref
    inds,_unused = signal.find_peaks(deltar, height = [-crossingRange, crossingRange]) #find peaks within +/- crossingRange of reference position 
    tcross = t[inds]
    rp = deltar[inds]
    
    #now we need to get rid of anything where there were big skips etc.
    #find the largest deviation, if it's within 2 standard deviations, we're done
    #otherwise, drop the point with the largest deviation and recalculate everything
    while True:
        deltat = np.gradient(tcross)
        dterr = np.abs(deltat - np.mean(deltat));
        if (np.max(dterr)) < 2*np.std(deltat):
            break
        ind = np.argmax(dterr)
        tcross = np.delete(tcross,ind)
        rp = np.delete(rp, ind)

    #fit the remaining data to a line, intercept is period at 0 rotation
    p = np.polyfit(rp, deltat, 1)   
    T = p[1] #period at reference location
    
    if (showPlots):
        plt.figure(1)
        plt.clf()
        plt.plot(t, x, t, xs)
        #zoom to region surrounding zero crossing
        ind = np.argmin(np.abs(rp))
        plt.xlim(tcross[ind] + T*np.array([-0.05,0.05]))
        plt.ylim([-3*crossingRange, 3*crossingRange])
        plt.title ('one pass, zoomed in')
        plt.ylabel('distance from reference mark (m)')
        plt.xlabel ('t (s)')
        plt.legend (['raw data', 'smoothed'])
        plt.show()
        
        plt.figure(2)
        plt.clf()
        plt.plot(t, xs, tcross, -rp,'r.')
        plt.ylim([-3*crossingRange, 3*crossingRange])
        plt.title ('peak locations')
        plt.ylabel('distance from reference mark (m)')
        plt.xlabel ('t (s)')
        plt.legend (['smoothed data', 'peak locations'])
        plt.show()

        plt.figure(3)
        plt.clf()
        plt.plot(rp, deltat, 'ro', rp, np.polyval(p,rp))
        plt.xlabel('dist from mark (m)')
        plt.ylabel('period (s)')
        plt.legend(['data', 'fit'])
        plt.show()
    
    return 2*np.pi/T
