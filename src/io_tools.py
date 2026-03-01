"""
Input-Output Analysis Tools

This module provides functions for loading and analyzing input-output matrices,
computing technical coefficients, Leontief inverses, and linkage measures.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_io_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load input-output matrix from Excel file.
    
    Parameters
    ----------
    path : Path
        Path to Excel file containing the IO matrix.
        Expected format:
        - Sheet name: "Hoja1"
        - First column: sector names (index)
        - Columns S1_Agro...S20_EduSalud: technical coefficients matrix Z (20x20)
        - Last column: "Output" with total output x (20x1)
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        Z : np.ndarray
            Technical matrix (n x n), flows between sectors.
        x : np.ndarray
            Total output vector (n,), positive values.
        sectors : list[str]
            List of n sector names.
    
    Raises
    ------
    ValueError
        If Z contains NaN values or x contains non-positive values.
    """
    df = pd.read_excel(path, sheet_name='Hoja1')
    
    # Extract sector names (from first column)
    sectors = df['Sector'].tolist()
    n = len(sectors)
    
    # Extract Z matrix (intermediate consumption: S1_Agro to S20_EduSalud)
    sector_cols = [col for col in df.columns if col.startswith('S') and col != 'Sector'][:n]
    Z = df[sector_cols].values.astype(float)
    
    # Extract output vector
    x = df['Output'].values.astype(float)
    
    # Validation
    if np.isnan(Z).any():
        raise ValueError("Z matrix contains NaN values")
    if np.any(x <= 0):
        raise ValueError("Output vector x must contain only positive values")
    
    return Z, x, sectors


def technical_coefficients(Z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute technical coefficients matrix A.
    
    Each element A_ij represents the amount of sector i's output needed
    to produce one unit of sector j's output.
    
    Parameters
    ----------
    Z : np.ndarray
        Intermediate consumption matrix (n x n).
    x : np.ndarray
        Total output vector (n,).
    
    Returns
    -------
    A : np.ndarray
        Technical coefficients matrix (n x n), A_ij = Z_ij / x_j.
    
    Raises
    ------
    ValueError
        If Z and x have incompatible dimensions.
    """
    if Z.shape[0] != Z.shape[1] or Z.shape[1] != len(x):
        raise ValueError("Z must be square matrix and x must have length equal to Z.shape[1]")
    
    # Divide columns of Z by corresponding elements of x
    A = Z / x  # Broadcasting: each column is divided by corresponding x element
    
    return A


def leontief_inverse(A: np.ndarray) -> np.ndarray:
    """
    Compute Leontief inverse: L = (I - A)^(-1).
    
    The Leontief inverse measures total (direct + indirect) requirements
    per unit of final demand.
    
    Parameters
    ----------
    A : np.ndarray
        Technical coefficients matrix (n x n).
    
    Returns
    -------
    L : np.ndarray
        Leontief inverse matrix (n x n), L = inv(I - A).
    
    Raises
    ------
    ValueError
        If (I - A) is singular or close to singular.
    """
    n = A.shape[0]
    I = np.eye(n)
    
    try:
        L = np.linalg.inv(I - A)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Cannot invert (I - A): matrix is singular or ill-conditioned. {str(e)}")
    
    return L


def backward_forward_direct(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute direct backward and forward linkages.
    
    Backward linkage: how much a sector demands from other sectors.
    Forward linkage: how much other sectors demand from a given sector.
    
    Parameters
    ----------
    A : np.ndarray
        Technical coefficients matrix (n x n).
    
    Returns
    -------
    backward_direct : np.ndarray
        Direct backward linkages (n,), sum of A over rows (column sums).
    forward_direct : np.ndarray
        Direct forward linkages (n,), sum of A over columns (row sums).
    """
    # Backward: sum of each column (how much each sector demands from others)
    backward_direct = A.sum(axis=0)
    
    # Forward: sum of each row (how much others demand from each sector)
    forward_direct = A.sum(axis=1)
    
    return backward_direct, forward_direct


def backward_forward_indirect(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute total (direct + indirect) backward and forward linkages.
    
    These are based on the Leontief inverse and capture all interdependencies.
    
    Parameters
    ----------
    L : np.ndarray
        Leontief inverse matrix (n x n).
    
    Returns
    -------
    backward_total : np.ndarray
        Total backward linkages (n,), sum of L over rows (column sums).
    forward_total : np.ndarray
        Total forward linkages (n,), sum of L over columns (row sums).
    """
    # Backward: sum of each column in L (total effects on a sector from final demand)
    backward_total = L.sum(axis=0)
    
    # Forward: sum of each row in L (total effects a sector has on others)
    forward_total = L.sum(axis=1)
    
    return backward_total, forward_total


def rank_linkages(values: np.ndarray, sectors: list[str], top: int = 10) -> pd.DataFrame:
    """
    Rank sectors by linkage values (backward or forward).
    
    Parameters
    ----------
    values : np.ndarray
        Array of linkage values (n,), one per sector.
    sectors : list[str]
        List of sector names (length n).
    top : int, optional
        Number of top sectors to return (default 10).
    
    Returns
    -------
    result : pd.DataFrame
        DataFrame with columns ['sector', 'value'] sorted descending by value.
        Number of rows = min(top, len(sectors)).
    """
    df = pd.DataFrame({'sector': sectors, 'value': values})
    df = df.sort_values('value', ascending=False).head(top).reset_index(drop=True)
    
    return df
