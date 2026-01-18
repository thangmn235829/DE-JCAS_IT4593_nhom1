import numpy as np
from scipy.signal import find_peaks


def findNulls(P, nrOfNulls=2):
    """
    Function returns the indices corresponding to the nulls of the pattern P.
    By default, two closest-to-the-main-lobe nulls will be found.

    Parameters:
        P: pattern magnitude
        nrOfNulls: number of nulls to find (default: 2)

    Returns:
        nullIdcs: indices of nulls
    """
    # Flip the pattern to find valleys
    peaks, _ = find_peaks(-P)

    # Find main lobe index (maximum of P)
    idxMax = np.argmax(P)

    # Sort peaks by distance to main lobe
    distances = np.abs(peaks - idxMax)
    sorted_indices = np.argsort(distances)

    # Take the closest nulls
    closest_null_indices = peaks[sorted_indices[:nrOfNulls]]
    nullIdcs = np.sort(closest_null_indices)

    return nullIdcs
