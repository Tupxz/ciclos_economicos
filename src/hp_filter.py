"""
Hodrick-Prescott Filter implementation using linear algebra.

This module provides a manual implementation of the HP filter without relying
on external HP filter libraries. The filter decomposes a time series into
trend and cyclical components using quadratic penalized optimization solved
via linear algebra.
"""

import numpy as np
from typing import Tuple, Union


def _build_second_diff_matrix(n: int) -> np.ndarray:
    """
    Build the second difference matrix D for size (n-2) x n.
    
    The second difference operator computes Δ²y_t = y_t - 2*y_{t-1} + y_{t-2}.
    This matrix D is constructed such that D @ y gives all second differences.
    
    For a series of length n, D is (n-2) x n with structure:
    [  1  -2   1   0   0  ...  0 ]
    [  0   1  -2   1   0  ...  0 ]
    [  0   0   1  -2   1  ...  0 ]
    [ ...                      ... ]
    [  0  ...  0   1  -2   1   0 ]
    [  0  ...  0   0   1  -2   1 ]
    
    Parameters
    ----------
    n : int
        Length of the time series.
        
    Returns
    -------
    D : np.ndarray
        Second difference matrix of shape (n-2, n).
    """
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1
        D[i, i + 1] = -2
        D[i, i + 2] = 1
    return D


def hp_manual(y: Union[np.ndarray, list], lam: float = 1600) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hodrick-Prescott Filter: manual implementation using linear algebra.
    
    Decomposes a time series y into trend (tau) and cyclical (c) components
    by solving the optimization problem:
    
        min_tau Σ (y_t - tau_t)² + λ Σ (Δ² tau_t)²
    
    This is solved as a linear system:
    
        (I + λ D'D) tau = y
    
    where D is the second difference matrix and I is the identity matrix.
    
    Note: This implementation uses dense matrices (numpy.linalg.solve) and is 
    suitable for moderately-sized series (e.g., typical quarterly economic data 
    with n up to a few thousand observations).
    
    Parameters
    ----------
    y : np.ndarray or list
        Time series data. Must be 1-dimensional, length >= 4, no NaNs.
    lam : float, optional
        Smoothing parameter (default: 1600 for quarterly data).
        Higher values produce smoother trends.
        
    Returns
    -------
    trend : np.ndarray
        Estimated trend component (tau).
    cycle : np.ndarray
        Estimated cyclical component (y - tau).
        
    Raises
    ------
    ValueError
        If y is not 1D, has length < 4, contains NaNs, or if lam <= 0.
        
    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([10, 11, 12, 11, 10, 9, 10, 11])
    >>> trend, cycle = hp_manual(y, lam=1600)
    >>> print(trend.shape, cycle.shape)
    (8,) (8,)
    """
    # Convert input to numpy array
    y = np.asarray(y, dtype=float)
    
    # Validate input
    if y.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
    
    n = len(y)
    if n < 4:
        raise ValueError(f"y must have length >= 4, got {n}")
    
    if np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")
    
    if lam <= 0:
        raise ValueError(f"lam must be positive, got {lam}")
    
    # Build second difference matrix D (shape: (n-2, n))
    D = _build_second_diff_matrix(n)
    
    # Compute D'D (shape: (n, n))
    DTD = D.T @ D
    
    # Construct the system matrix: I + λ D'D
    I = np.eye(n)
    A = I + lam * DTD
    
    # Solve the linear system: A tau = y
    # Using numpy.linalg.solve for numerical stability
    trend = np.linalg.solve(A, y)
    
    # Extract the cyclical component
    cycle = y - trend
    
    return trend, cycle


def gap_ratio(y: Union[np.ndarray, list], trend: Union[np.ndarray, list]) -> np.ndarray:
    """
    Compute the output gap as a ratio: (y / trend) - 1.
    
    This represents the percentage deviation of the series from its trend.
    Useful for economic analysis where we want cyclical deviations as
    a proportion of the trend level.
    
    Parameters
    ----------
    y : np.ndarray or list
        Actual series values.
    trend : np.ndarray or list
        Trend estimates (typically from hp_manual).
        
    Returns
    -------
    gap : np.ndarray
        Output gap ratio (y / trend) - 1.
        Positive values indicate above-trend, negative indicate below-trend.
        
    Raises
    ------
    ValueError
        If y and trend have different lengths.
    ZeroDivisionError (implicit)
        If trend contains zeros (raises division warning/error).
        
    Examples
    --------
    >>> y = np.array([100, 102, 101, 99])
    >>> trend = np.array([100, 100.5, 101, 100.5])
    >>> gap = gap_ratio(y, trend)
    >>> print(gap)
    [ 0.    0.01492537  -0.00990099  -0.00497512]
    """
    y = np.asarray(y, dtype=float)
    trend = np.asarray(trend, dtype=float)
    
    if len(y) != len(trend):
        raise ValueError(f"y and trend must have same length, got {len(y)} and {len(trend)}")
    
    if np.any(trend == 0):
        raise ValueError("trend contains zero values; cannot compute gap ratio (division by zero)")
    
    gap = (y / trend) - 1
    
    return gap
