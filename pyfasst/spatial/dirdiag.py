'''dirdiag.py

Draw directivity diagrams

2013 Jean-Louis Durrieu
http://www.durrieu.ch
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax3d
import numpy as np
from numpy.linalg import inv

from .. import audioModel as am
from ..tools import signalTools as st
from ..tools.utils import db, ident

sound_celerity = 300. # m/s

def make_MVDR_filter_target(steering_vec_target, steering_vec_interf):
    '''make MVDR spatial filter from estimated steering vectors
    '''
    # get the sizes:
    n_target, n_freqs, n_chans = steering_vec_target.shape
    n_interf, n_freq2, n_chan2 = steering_vec_interf.shape
    
    if n_freqs!=n_freq2 or n_chans!=n_chan2:
        raise ValueError("Not the right dim for the steering_vectors")
    
    if n_chans!=2:
        raise AttributeError("Program only for stereo filters.")
    
    if n_target!=1:
        raise AttributeError("For now, only for rank-1 spatial target source.")
    
    RaaDiag0 = (
        np.abs(steering_vec_target[0,:,0])**2
        + np.sum(np.abs(steering_vec_interf[:,:,0])**2, axis=0)
        )
    RaaDiag1 = (
        np.abs(steering_vec_target[0,:,1])**2
        + np.sum(np.abs(steering_vec_interf[:,:,1])**2, axis=0)
        )
    RaaOff = (
        (steering_vec_target[0,:,0] *
         np.conjugate(steering_vec_target[0,:,1]))
        + np.sum(steering_vec_interf[:,:,0] *
                 np.conjugate(steering_vec_interf[:,:,1]),
                 axis=0)
        )
    invRaa_00, invRaa_01, invRaa_11 = st.invHermMat2D(
        RaaDiag0, RaaOff, RaaDiag1)
    
    w_filter = np.zeros([n_chans, n_freqs],
                        dtype=np.complex)
    w_filter[0] = (
        invRaa_00 * steering_vec_target[0,:,0] +
        invRaa_01 * steering_vec_target[0,:,1]
        )
    w_filter[1] = (
        np.conjugate(invRaa_01) * steering_vec_target[0,:,0] +
        invRaa_11 * steering_vec_target[0,:,1]
        )
    w_filter /= (
        w_filter[0] * np.conjugate(steering_vec_target[0,:,0]) +
        w_filter[1] * np.conjugate(steering_vec_target[0,:,1])
        )
    w_filter = np.conjugate(w_filter)
    return w_filter

def directivity_filter_diagram_ULA(n_sensors=2, dist_inter_sensor=.15,
                                   w_filter=None, theta_filter=0,
                                   freqs=None, thetas=None,
                                   nfreqs=256, nthetas=30,
                                   samplerate=8000., doplot='2d',
                                   fig=None, subplot_number=(1,1,1),
                                   dyn_func=db):
    '''Computes and displays the directivity diagram associated to
    the provided filter ``w_filter`` (optionally parameterized by the
    targetted direction ``theta_filter``).
    
    the diagram can be interpreted as the amplitude of the filter for
    a source at frequency ``f``, located at an angle ``theta`` from
    the the ULA array axis.
    '''
    
    # setting the frequency range
    if freqs is None:
        freqs = np.arange(nfreqs) * samplerate / (2. * nfreqs)
    else:
        nfreqs = len(freqs)
        
    # setting the theta range
    if thetas is None:
        thetas = np.arange(-nthetas, nthetas) * np.pi / (2. * nthetas)
    
    nthetas = len(thetas)
    
    # initializing the filter as a steering vector in direction of theta:
    if w_filter is None:
        w_filter = np.conjugate(am.gen_steer_vec_far_src_uniform_linear_array(
            freqs=freqs,
            nchannels=n_sensors,
            theta=theta_filter,
            distanceInterMic=dist_inter_sensor)) / np.sqrt(n_sensors)
    else: # check that the length is correct
        if w_filter.shape != (n_sensors, nfreqs):
            print "w_filter.shape", w_filter.shape
            print "expected shape", (n_sensors, nfreqs)
            raise ValueError("w_filter does not have the correct shape.")
        
    # computing the matrix of diagram:
    diagram = np.zeros([nthetas, nfreqs])
    for n_theta, theta in enumerate(thetas):
        a = am.gen_steer_vec_far_src_uniform_linear_array(
            freqs=freqs,
            nchannels=n_sensors,
            theta=theta,
            distanceInterMic=dist_inter_sensor) / np.sqrt(n_sensors)
        diagram[n_theta] = np.abs(
            (w_filter*a).sum(axis=0))**2
        
    # displayin
    if doplot == '3d':
        if fig is None:
            fig=plt.figure()
        ax = ax3d(fig)
        ax.plot_surface(np.outer(thetas, np.ones(nfreqs)),
                        np.outer(np.ones(nthetas), freqs),
                        dyn_func(diagram),linewidth=0, 
                        rstride=1, cstride=1,
                        cmap=plt.cm.gray)
    elif doplot == '2d':
        if fig is None:
            fig=plt.figure()
        ax = fig.add_subplot(*subplot_number)
        ax.imshow(dyn_func(diagram))
        plt.xlabel('frequencies')
        plt.ylabel('$\\theta$ (in $\pi$)')
        # for now, this only works for freqs linear, from 0 to fs/2:
        nlabs = 4
        xlabs = (
            np.arange(1,nlabs+1) / (2.* (nlabs + 2.)) * samplerate / 1000.)
        xpos = np.arange(1,nlabs+1) / (nlabs + 2.) * nfreqs
        plt.xticks(xpos, ['%.2f' %lab for lab in xlabs])
        ypos, ylab = plt.yticks()
        ypos = np.int32(ypos[1:-1])
        ylab = thetas[ypos] / np.pi
        ylab = ['%.2f' %lab for lab in ylab]
        plt.yticks(ypos, ylab)
        
    plt.draw()
        
    return thetas, freqs, diagram

def producePicDiagramAgainstDistNSensors(w_filter=None,
                                         theta_filter=np.pi/4.,
                                         sensors=None, dists=None,
                                         samplerate=8000.,
                                         doplot='2d', dyn_func=db,
                                         thetas=None, nthetas=30):
    """generate a drawing that shows the directivity diagrams for
    several values of distance between sensors and number of sensors.
    """
    
    if sensors is None:
        sensors = [2, 5, 10]
    if dists is None:
        dists = [0.1, 0.15, 0.2, 0.5]
        
    fig=plt.figure(figsize=(3, 5))
    
    for nd, d in enumerate(dists):
        for ns, s in enumerate(sensors):
            truc = directivity_filter_diagram_ULA(
                w_filter=w_filter,
                theta_filter=theta_filter,
                n_sensors=s,
                samplerate=samplerate,
                nthetas=nthetas, thetas=thetas,
                dist_inter_sensor=d,
                fig=fig,doplot=doplot,
                subplot_number=(len(dists),
                                len(sensors),
                                1+len(sensors)*nd+ns),
                dyn_func=dyn_func)
                               #int('%d%d%d' %(len(dists),
                               #               len(sensors),
                               #               1+len(sensors)*ns+nd)))
            # show the zone where there is no ambiguity with the angle:
            if theta_filter==0:
                lim1 = np.arcsin(-sound_celerity/(2*truc[1]*d))
                lim2 = np.arcsin(sound_celerity/(2*truc[1]*d))
                lim1 *= (2. * 30 / np.pi)
                lim1 += 30
                lim2 *= (2. * 30 / np.pi)
                lim2 += 30
                plt.plot(lim1, '--k', label='valid $\\theta$')
                plt.plot(lim2, '--k', label='valid $\\theta$')
                # plt.legend()
            plt.title('%f %d' %( d, s) )
        
    plt.subplots_adjust(hspace=0.5, wspace=0.2,
                        left=0.05, right=0.98,top=0.95, bottom=0.08)
    return truc

def generate_steer_vec_thetas(n_sensors=2, dist_inter_sensors=0.15,
                              freqs=None, n_freqs=256,
                              thetas=None, n_thetas=30,
                              samplerate=8000., computeRaa=False,
                              ):
    '''Generates a collection of steering vectors with the provided array and
    signal parameters.
    
    '''
    if freqs is None:
        freqs = np.arange(n_freqs) * samplerate / (2. * n_freqs)
    else:
        n_freqs = len(freqs)
        
    # setting the theta range
    if thetas is None:
        thetas = np.arange(-n_thetas, n_thetas) * np.pi / (2. * n_thetas)
    
    n_thetas = len(thetas)
    
    A = np.zeros([n_freqs, n_sensors, n_thetas],
                 dtype=np.complex)
    for ntheta, theta in enumerate(thetas):
        A[:,:,ntheta] = (
            am.gen_steer_vec_far_src_uniform_linear_array(
                freqs=freqs,
                nchannels=n_sensors,
                theta=theta,
                distanceInterMic=dist_inter_sensors).T) / np.sqrt(n_sensors)
    
    if computeRaa:
        Raa = np.zeros([n_freqs, n_sensors, n_sensors],
                       dtype=np.complex)
        for nfreq, freq in enumerate(freqs):
            Raa[nfreq] = np.dot(A[nfreq], np.conjugate(A[nfreq]).T)
        return freqs, thetas, A, Raa
    return freqs, thetas, A

def produceMVDRplots(theta_filter=np.pi/4., freqs=None, n_freqs=256,
                     thetas=None, n_thetas=30, samplerate=8000.,
                     n_sensors=2, dist_inter_sensors=0.15,
                     dists=[0.15, 0.5, 1.], doplot='2d', dyn_func=db,
                     theta_interf=None, n_theta_interf=4):
    '''MVDR gains for spatial filtering

    **Description**:
    
    Minimum Variance - Distortionless Response
    
    **Examples**::

     >>># importing the module
     >>>import pyfasst.spatial.dirdiag as dd
     >>># the MVDR plots, for 4 sensors and 4 interferers: visible rejection
     >>>Raa, w, th, fr, thetas, diag = dd.produceMVDRplots(n_sensors=4, 
            n_theta_interf=2, samplerate=8000., dists=[0.15,])
     >>># plotting also the filter responses against the angles
     >>>plt.figure();plt.plot(thetas,dd.db(diag))
     >>>plt.xlabel('$\\theta$ (rad)');plt.ylabel('Response (dB)')
     >>># MVDR plots 2 sensors for 4 interferers: no rejection!
     >>>Raa, w, th, fr, thetas, diag = dd.produceMVDRplots(n_sensors=2,
            n_theta_interf=2, samplerate=8000., dists=[0.15,])
     >>># plotting also the filter responses against the angles
     >>>plt.figure();plt.plot(thetas,dd.db(diag))
     >>>plt.xlabel('$\\theta$ (rad)');plt.ylabel('Response (dB)')
    
    '''
    freqs, theta_interf, steering_vectors, Raa = generate_steer_vec_thetas(
        n_sensors=n_sensors, dist_inter_sensors=dist_inter_sensors,
        freqs=freqs, n_freqs=n_freqs,
        thetas=theta_interf, n_thetas=n_theta_interf,
        samplerate=samplerate, computeRaa=True
        )
    n_freqs = len(freqs)
    
    w_filter = np.ones([n_sensors, n_freqs],
                        dtype=np.complex) / np.sqrt(n_sensors)
    
    if any(np.abs((theta_filter-theta_interf)**2)<1e-3):#theta_filter in thetas:
        print "we re here"
        err = np.abs((theta_filter-theta_interf)**2)
        i_theta = np.argmin(err)
        print "taking", theta_interf[i_theta] 
        target_loc_vec = steering_vectors[:,:,i_theta].T
    else:
        target_loc_vec = (
            am.gen_steer_vec_far_src_uniform_linear_array(
                freqs=freqs,
                nchannels=n_sensors,
                theta=theta_filter,
                distanceInterMic=dist_inter_sensors)) / np.sqrt(n_sensors)

    # adding the target in Raa:
    for nf, f in enumerate(freqs):
        Raa[nf] += np.outer(target_loc_vec[:, nf],
                            np.conjugate(target_loc_vec[:, nf]))
    
    for nf, f in enumerate(freqs):
        if nf > 0: # avoiding freq 0, because Raa non-invertible
            w_filter[:, nf] = np.dot(inv(Raa[nf]), target_loc_vec[:, nf])
            w_filter[:, nf] /= np.dot(w_filter[:, nf],
                                      np.conjugate(target_loc_vec[:, nf]))
    w_filter[:] = np.conjugate(w_filter)
    
    thetas, freqs, diagram = producePicDiagramAgainstDistNSensors(
        w_filter=w_filter, sensors=[n_sensors,], dists=dists,
        thetas=thetas, nthetas=n_thetas,
        samplerate=samplerate, doplot=doplot, dyn_func=dyn_func)
    
    # displaying the lines for each of the interferers:
    for ax in plt.gcf().get_axes():
        ax.plot(np.outer(np.ones(n_freqs),
                          theta_interf) * (2. * n_thetas / np.pi)
                + n_thetas,
                 '--k', label='interferers')
        ax.plot(np.ones(n_freqs) * theta_filter * (2. * n_thetas / np.pi)
                + n_thetas,
                '--w', label='target')
    plt.draw()
    
    return Raa, w_filter, theta_interf, freqs, thetas, diagram
