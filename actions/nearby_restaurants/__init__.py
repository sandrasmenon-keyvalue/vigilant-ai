"""
Restaurant Finder Module

This module provides functionality to find nearby restaurants using OpenStreetMap's 
Overpass API. Fetches and returns restaurant data immediately without storing.
"""

from .restaurant_finder import RestaurantFinder

__all__ = ['RestaurantFinder']
__version__ = '2.0.0'
