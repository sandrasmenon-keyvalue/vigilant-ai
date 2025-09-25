"""
Nearby Restaurants Module

This module provides functionality to fetch, store, and retrieve nearby restaurants
using OpenStreetMap's Overpass API.
"""

from .store_nearby_restaurants import NearbyRestaurantsStore

__all__ = ['NearbyRestaurantsStore']
__version__ = '1.0.0'
