"""
Restaurant Finder using OpenStreetMap Data

This module provides functionality to find nearby restaurants using OpenStreetMap's 
Overpass API. Fetches and returns restaurant data immediately without storing.
"""

import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RestaurantFinder:
    """
    A class to find nearby restaurants using OpenStreetMap data.
    Fetches and returns restaurant data immediately without storing.
    """
    
    def __init__(self, radius_meters: int = 1000):
        """
        Initialize the restaurant finder.
        
        Args:
            radius_meters: Search radius in meters (default: 1000m = 1km)
        """
        self._radius_meters: int = radius_meters
        self._overpass_url: str = "https://overpass-api.de/api/interpreter"
        self._timeout_seconds: int = 30
        
        logger.info(f"Initialized RestaurantFinder with {radius_meters}m radius")
    
    def find_nearby_restaurants(self, latitude: float, longitude: float) -> Dict:
        """
        Find and return nearby restaurants immediately.
        
        Args:
            latitude: Latitude coordinate (-90 to 90)
            longitude: Longitude coordinate (-180 to 180)
            
        Returns:
            Dict containing:
            - 'restaurants': List of restaurant data
            - 'location': {'latitude': float, 'longitude': float} or None
            - 'last_updated': datetime or None
            - 'count': int
            - 'search_radius_meters': int
        """
        try:
            # Validate coordinates
            if not self._validate_coordinates(latitude, longitude):
                logger.error(f"Invalid coordinates: lat={latitude}, lon={longitude}")
                return {
                    'restaurants': [],
                    'location': None,
                    'last_updated': None,
                    'count': 0,
                    'search_radius_meters': self._radius_meters
                }
            
            logger.info(f"Finding restaurants near lat={latitude}, lon={longitude}")
            
            # Fetch restaurant data
            restaurants = self._fetch_restaurants_from_osm(latitude, longitude)
            last_updated = datetime.now()
            
            logger.info(f"Found {len(restaurants)} restaurants")
            
            # Return results immediately
            return {
                'restaurants': restaurants,
                'location': {
                    'latitude': latitude,
                    'longitude': longitude
                },
                'last_updated': last_updated,
                'count': len(restaurants),
                'search_radius_meters': self._radius_meters
            }
            
        except Exception as e:
            logger.error(f"Failed to find restaurants: {str(e)}")
            return {
                'restaurants': [],
                'location': None,
                'last_updated': None,
                'count': 0,
                'search_radius_meters': self._radius_meters
            }
    
    def _fetch_restaurants_from_osm(self, lat: float, lon: float) -> List[Dict]:
        """
        Private method to fetch restaurants from OpenStreetMap Overpass API.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            List of restaurant dictionaries with standardized format
        """
        try:
            # Build the Overpass query
            query = self._build_overpass_query(lat, lon)
            
            # Make the API request
            response = requests.post(
                self._overpass_url,
                data=query,
                timeout=self._timeout_seconds,
                headers={'Content-Type': 'text/plain; charset=utf-8'}
            )
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Extract and parse restaurant data
            restaurants = []
            for element in data.get('elements', []):
                restaurant = self._parse_restaurant_data(element)
                if restaurant:
                    restaurants.append(restaurant)
            
            logger.info(f"Fetched {len(restaurants)} restaurants from OSM")
            return restaurants
            
        except requests.exceptions.Timeout:
            logger.error("Timeout while fetching data from Overpass API")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching restaurants: {str(e)}")
            return []
    
    def _build_overpass_query(self, lat: float, lon: float) -> str:
        """
        Build the Overpass API query string for restaurants.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            str: Overpass QL query string
        """
        query = f"""
        [out:json][timeout:25];
        (
          node["amenity"~"^(restaurant|cafe|fast_food|bar|pub|food_court)$"](around:{self._radius_meters},{lat},{lon});
          way["amenity"~"^(restaurant|cafe|fast_food|bar|pub|food_court)$"](around:{self._radius_meters},{lat},{lon});
          relation["amenity"~"^(restaurant|cafe|fast_food|bar|pub|food_court)$"](around:{self._radius_meters},{lat},{lon});
        );
        out center meta;
        """
        return query.strip()
    
    def _parse_restaurant_data(self, osm_data: Dict) -> Optional[Dict]:
        """
        Parse OSM data into simplified restaurant format.
        
        Args:
            osm_data: Raw OSM element data
            
        Returns:
            Dict with simplified restaurant info or None if invalid data:
            - 'name': str - Restaurant name
            - 'link': str - Google Maps link with DMS coordinates
        """
        try:
            tags = osm_data.get('tags', {})
            
            # Skip if no name
            name = tags.get('name', '').strip()
            if not name:
                return None
            
            # Get coordinates
            if osm_data['type'] == 'node':
                lat = osm_data.get('lat')
                lon = osm_data.get('lon')
            else:
                # For ways and relations, use center coordinates
                center = osm_data.get('center', {})
                lat = center.get('lat')
                lon = center.get('lon')
            
            if lat is None or lon is None:
                return None
            
            # Build simplified restaurant data - only name and Google Maps link
            restaurant = {
                'name': name,
                'link': self._generate_google_maps_link(float(lat), float(lon))
            }
            
            return restaurant
            
        except Exception as e:
            logger.warning(f"Error parsing restaurant data: {str(e)}")
            return None
    
    def _generate_google_maps_link(self, latitude: float, longitude: float) -> str:
        """
        Generate Google Maps link using degrees, minutes, seconds format.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            
        Returns:
            str: Google Maps URL with DMS coordinates
        """
        def decimal_to_dms(decimal_degrees: float, is_longitude: bool = False) -> str:
            """Convert decimal degrees to degrees, minutes, seconds format."""
            # Determine direction
            if is_longitude:
                direction = 'E' if decimal_degrees >= 0 else 'W'
            else:
                direction = 'N' if decimal_degrees >= 0 else 'S'
            
            # Work with absolute value
            abs_degrees = abs(decimal_degrees)
            
            # Extract degrees, minutes, seconds
            degrees = int(abs_degrees)
            minutes_float = (abs_degrees - degrees) * 60
            minutes = int(minutes_float)
            seconds = (minutes_float - minutes) * 60
            
            # Format the string
            return f"{degrees}°{minutes:02d}'{seconds:04.1f}\"{direction}"
        
        # Convert coordinates to DMS format
        lat_dms = decimal_to_dms(latitude, False)
        lon_dms = decimal_to_dms(longitude, True)
        
        # Create the Google Maps URL
        return f"https://www.google.com/maps/place/{lat_dms}+{lon_dms}"
    
    def _validate_coordinates(self, latitude: float, longitude: float) -> bool:
        """
        Validate latitude and longitude coordinates.
        
        Args:
            latitude: Latitude value
            longitude: Longitude value
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            lat = float(latitude)
            lon = float(longitude)
            
            if not (-90 <= lat <= 90):
                return False
            if not (-180 <= lon <= 180):
                return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    
    def set_radius(self, radius_meters: int) -> bool:
        """
        Update the search radius.
        
        Args:
            radius_meters: New search radius in meters
            
        Returns:
            bool: True if radius was updated successfully
        """
        if radius_meters <= 0 or radius_meters > 50000:  # Max 50km
            logger.error(f"Invalid radius: {radius_meters}m (must be 1-50000)")
            return False
        
        self._radius_meters = radius_meters
        logger.info(f"Updated search radius to {radius_meters}m")
        return True


# Example usage and testing functions
def example_usage():
    """Example of how to use the RestaurantFinder class."""
    
    # Initialize the finder
    restaurant_finder = RestaurantFinder(radius_meters=1500)
    
    # Example coordinates (Times Square, NYC)
    latitude = 40.7589
    longitude = -73.9851
    
    print(f"Searching for restaurants near {latitude}, {longitude}")
    
    # Find nearby restaurants
    result = restaurant_finder.find_nearby_restaurants(latitude, longitude)
    
    if result['count'] > 0:
        print(f"\nFound {result['count']} restaurants:")
        print(f"Search location: {result['location']}")
        print(f"Last updated: {result['last_updated']}")
        print(f"Search radius: {result['search_radius_meters']}m")
        
        # Display first 5 restaurants
        for i, restaurant in enumerate(result['restaurants'][:5]):
            print(f"\n{i+1}. {restaurant['name']}")
            print(f"   Google Maps: {restaurant['link']}")
        
    else:
        print("No restaurants found or search failed")


if __name__ == "__main__":
    example_usage()
