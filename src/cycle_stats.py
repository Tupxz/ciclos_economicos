"""
Cycle statistics and stylized facts using statsmodels HP filter and correlation analysis.

Module for computing output gaps, volatility tables, and correlation matrices
from HP-decomposed macroeconomic series.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter


def hp_library_gap(series, lam=1600):
    """
    Compute HP filter gap using statsmodels hpfilter.
    
    Parameters
    ----------
    series : pd.Series
        Time series in levels with datetime index.
    lam : int, default=1600
        HP smoothness parameter (1600 for quarterly data).
    
    Returns
    -------
    pd.Series
        Output gap defined as (y / trend) - 1, with same index and name
        constructed as f"{series.name}_gap" or "series_gap" if name is None.
    
    Raises
    ------
    ValueError
        If trend contains zero values (division by zero in gap calculation).
    
    Notes
    -----
    - Input series with NaN values will be dropped before filtering.
    - The HP filter from statsmodels returns (cycle, trend) where trend
      is the long-run component.
    """
    # Drop NaN values from input series
    series_clean = series.dropna()
    
    # Apply HP filter using statsmodels
    cycle, trend = hpfilter(series_clean, lamb=lam)
    
    # Check for zero values in trend (would cause division by zero)
    if np.any(trend == 0):
        raise ValueError("Trend contains zero values; cannot compute gap ratio (division by zero).")
    
    # Compute gap as (y / trend) - 1
    gap = (series_clean / trend) - 1
    
    # Construct series name
    gap_name = f"{series.name}_gap" if series.name else "series_gap"
    gap.name = gap_name
    
    return gap


def volatility_table(gaps_df, gdp_col="GDP"):
    """
    Compute volatility (std dev) and relative volatility with respect to GDP.
    
    Parameters
    ----------
    gaps_df : pd.DataFrame
        DataFrame with gap columns.
    gdp_col : str, default="GDP"
        Column name for GDP gap (used as reference for relative volatility).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - std: standard deviation of each gap
        - std_rel_to_gdp: std(gap) / std(GDP_gap)
    
    Raises
    ------
    ValueError
        If gdp_col not present in gaps_df.
    """
    if gdp_col not in gaps_df.columns:
        raise ValueError(f"Column '{gdp_col}' not found in gaps_df.")
    
    # Compute standard deviations
    std_devs = gaps_df.std()
    
    # Compute relative volatility (ratio to GDP std)
    gdp_std = gaps_df[gdp_col].std()
    std_rel = std_devs / gdp_std
    
    # Create result DataFrame
    result = pd.DataFrame({
        'std': std_devs,
        'std_rel_to_gdp': std_rel
    })
    
    return result


def corr_table(gaps_df, gdp_col="GDP"):
    """
    Compute contemporaneous correlations with GDP gap (excluding GDP self-correlation).
    
    Parameters
    ----------
    gaps_df : pd.DataFrame
        DataFrame with gap columns.
    gdp_col : str, default="GDP"
        Column name for GDP gap.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with column corr_with_gdp: correlation of each gap with GDP gap.
        GDP self-correlation is excluded from the result.
    
    Raises
    ------
    ValueError
        If gdp_col not present in gaps_df.
    """
    if gdp_col not in gaps_df.columns:
        raise ValueError(f"Column '{gdp_col}' not found in gaps_df.")
    
    # Compute correlations with GDP gap
    gdp_gap = gaps_df[gdp_col]
    correlations = gaps_df.corrwith(gdp_gap)
    
    # Remove GDP self-correlation
    correlations = correlations.drop(gdp_col, errors='ignore')
    
    # Create result DataFrame
    result = pd.DataFrame({
        'corr_with_gdp': correlations
    })
    
    return result


def rolling_corr_table(gaps_df, window, gdp_col="GDP"):
    """
    Compute rolling correlations with GDP gap.
    
    Parameters
    ----------
    gaps_df : pd.DataFrame
        DataFrame with gap columns and datetime index.
    window : int
        Rolling window size (must be >= 4).
    gdp_col : str, default="GDP"
        Column name for GDP gap.
    
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with columns corr_<var>_gdp for each variable.
        First (window - 1) rows will be NaN due to rolling window initialization.
    
    Raises
    ------
    ValueError
        If window < 4 or gdp_col not present in gaps_df.
    
    Notes
    -----
    NaN values at the beginning of the result are expected and reflect
    the minimum window size needed to compute rolling correlation.
    """
    if window < 4:
        raise ValueError(f"Window size must be >= 4, got {window}.")
    
    if gdp_col not in gaps_df.columns:
        raise ValueError(f"Column '{gdp_col}' not found in gaps_df.")
    
    # Get GDP gap series
    gdp_gap = gaps_df[gdp_col]
    
    # Compute rolling correlation for each column
    rolling_corrs = {}
    for col in gaps_df.columns:
        if col != gdp_col:
            rolling_corr = gaps_df[col].rolling(window=window).corr(gdp_gap)
            rolling_corrs[f"corr_{col}_gdp"] = rolling_corr
    
    # Create result DataFrame
    result = pd.DataFrame(rolling_corrs, index=gaps_df.index)
    
    return result
