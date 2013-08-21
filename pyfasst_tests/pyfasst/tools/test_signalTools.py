"""tests for pyfasst.tools.signalTools

2013 Jean-Louis Durrieu
"""

from ...testing import * # is this really legal?

import numpy as np
import pyfasst.tools.signalTools as st

def test_medianFilter():
    """median filtering an all 0 array, with one different value in the middle
    """
    inputArray = np.zeros(100)
    inputArray[50] = 2
    outputArray = st.medianFilter(inputArray)
    assert_array_equal(outputArray, np.zeros(100))

def test_medianFilter_withNaN():
    """median filtering an all 0 array, with one NaN value
    """
    inputArray = np.zeros(100)
    inputArray[50] = np.NaN
    outputArray = st.medianFilter(inputArray)
    assert_array_equal(outputArray, np.zeros(100))

# some Hermitian matrix coming from a run of audioModel.multichanLead
sigma_x_diag = np.array(
    [[ 0.00977917,  0.01021195,  0.00949931,  0.01081156,  0.00982221,
       0.00927985,  0.01090643,  0.01078789,  0.00941831,  0.01113587],
     [ 0.00785231,  0.00819886,  0.00762822,  0.00867899,  0.00788678,
       0.00745249,  0.00875495,  0.00866003,  0.00756336,  0.00893867]])
sigma_x_off = np.array(
    [ 0.00865282+0.j,  0.00904009+0.j,  0.00840240+0.j,  0.00957665+0.j,
      0.00869134+0.j,  0.00820601+0.j,  0.00966154+0.j,  0.00955547+0.j,
      0.00832991+0.j,  0.00986685+0.j])
inv_sigma_x_diag_ref = np.array(
    [[ 4094.58407492,  4093.23448666,  4095.52259353,  4091.54400714,
       4094.44450083,  4096.2983896 ,  4091.29365679,  4091.60715192,
       4095.80470268,  4090.70587309],
     [ 5099.34077392,  5098.26012193,  5100.09227127,  5096.90650902,
       5099.22901315,  5100.71347227,  5096.7060467 ,  5096.95707075,
       5100.31816375,  5096.2353923 ]])
inv_sigma_x_off_ref = np.array(
    [-4512.00955238+0.j, -4513.21720930+0.j, -4511.16973437+0.j,
     -4514.72990713+0.j, -4512.13444795+0.j, -4510.47552598+0.j,
     -4514.95392908+0.j, -4514.67340307+0.j, -4510.91729363+0.j,
     -4515.47989766+0.j])

def test_inv_herm_mat_2d():
    """invert a 2D Hermitian matrix
    """
    inv_sigma_x_diag, inv_sigma_x_off, det_sigma_x = (
        st.inv_herm_mat_2d(sigma_x_diag, sigma_x_off))
    assert_array_almost_equal(
        inv_sigma_x_diag[0] * sigma_x_diag[0] +
        sigma_x_off * np.conj(inv_sigma_x_off),
        np.ones_like(inv_sigma_x_off)
        )
    assert_array_almost_equal(
        inv_sigma_x_diag[0] * np.conj(sigma_x_off) +
        sigma_x_diag[1] * np.conj(inv_sigma_x_off),
        np.zeros_like(inv_sigma_x_off)
        )

