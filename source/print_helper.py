import numpy as np
import pandas as pd


def format_point(pt):
    """
    Formats a point (list of coordinates) into a string representation for easy upload
    into the BBO submission portal.

    The point is converted to a string where each coordinate is formatted to 6 
    d.p. and joined by hyphens.

    Args:
        pt (list): The coordinates to format. 

    Returns:
        str: A formatted string (e.g. '0.123456-0.987654').
    """
    
    if pt is None:
        return ""    

    return '-'.join([f'{p:.6f}' for p in pt])