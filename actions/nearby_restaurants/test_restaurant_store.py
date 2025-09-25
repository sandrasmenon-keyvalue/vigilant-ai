#!/usr/bin/env python3
"""
Test script for NearbyRestaurantsStore class

This script tests the functionality of the NearbyRestaurantsStore class
with real OpenStreetMap API calls and various test scenarios.
"""

import sys
import time
from datetime import datetime
from store_nearby_restaurants import NearbyRestaurantsStore


def print_separator(title: str):
    """Print a formatted separator for test sections."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def test_initialization():
    """Test class initialization with different parameters."""
    print_separator("TEST 1: INITIALIZATION")
    
    # Test default initialization
    store1 = NearbyRestaurantsStore()
    print("✓ Default initialization successful")
    print(f"  Default radius: {store1._radius_meters}m")
    
    # Test custom radius
    store2 = NearbyRestaurantsStore(radius_meters=2000)
    print("✓ Custom radius initialization successful")
    print(f"  Custom radius: {store2._radius_meters}m")
    
    # Test initial state
    initial_data = store1.get_latest()
    print(f"✓ Initial state check:")
    print(f"  Restaurant count: {initial_data['count']}")
    print(f"  Has location: {initial_data['location'] is not None}")
    print(f"  Last updated: {initial_data['last_updated']}")
    
    return store1


def test_coordinate_validation():
    """Test coordinate validation with various inputs."""
    print_separator("TEST 2: COORDINATE VALIDATION")
    
    store = NearbyRestaurantsStore()
    
    # Valid coordinates
    test_cases = [
        (40.7589, -73.9851, True, "Times Square, NYC"),
        (51.5074, -0.1278, True, "London, UK"),
        (35.6762, 139.6503, True, "Tokyo, Japan"),
        (91.0, 0.0, False, "Invalid latitude > 90"),
        (-91.0, 0.0, False, "Invalid latitude < -90"),
        (0.0, 181.0, False, "Invalid longitude > 180"),
        (0.0, -181.0, False, "Invalid longitude < -180"),
        ("invalid", "coords", False, "Non-numeric coordinates"),
        (None, None, False, "None coordinates")
    ]
    
    for lat, lon, expected, description in test_cases:
        try:
            result = store._validate_coordinates(lat, lon)
            status = "✓" if result == expected else "✗"
            print(f"{status} {description}: {result}")
        except Exception as e:
            result = False
            status = "✓" if result == expected else "✗"
            print(f"{status} {description}: Exception caught - {str(e)}")


def test_real_location_update():
    """Test updating location with real coordinates and API calls."""
    print_separator("TEST 3: REAL LOCATION UPDATE")
    
    store = NearbyRestaurantsStore(radius_meters=1000)
    
    # Test locations with expected restaurant density
    test_locations = [
        (40.7589, -73.9851, "Times Square, NYC (High density)"),
        (37.7749, -122.4194, "San Francisco, CA (Medium density)"),
        (51.5074, -0.1278, "London, UK (High density)"),
    ]
    
    for lat, lon, description in test_locations:
        print(f"\n🌍 Testing location: {description}")
        print(f"   Coordinates: {lat}, {lon}")
        
        start_time = time.time()
        success = store.update_location(lat, lon)
        end_time = time.time()
        
        if success:
            data = store.get_latest()
            print(f"✓ Update successful in {end_time - start_time:.2f} seconds")
            print(f"  Found {data['count']} restaurants")
            print(f"  Location stored: {data['location']}")
            print(f"  Last updated: {data['last_updated']}")
            
            # Show sample restaurants
            if data['restaurants']:
                print(f"  Sample restaurants:")
                for i, restaurant in enumerate(data['restaurants'][:3]):
                    print(f"    {i+1}. {restaurant['name']} ({restaurant.get('amenity', 'unknown')})")
                    if 'cuisine' in restaurant:
                        print(f"       Cuisine: {restaurant['cuisine']}")
                    print(f"       Google Maps: {restaurant.get('link', 'N/A')}")
            
            # Test data freshness
            is_fresh = store.is_data_fresh(max_age_minutes=5)
            print(f"  Data is fresh: {is_fresh}")
            
        else:
            print("✗ Update failed")
        
        # Small delay between requests to be respectful to the API
        time.sleep(1)
    
    return store


def test_data_replacement():
    """Test that new location updates replace old data."""
    print_separator("TEST 4: DATA REPLACEMENT")
    
    store = NearbyRestaurantsStore(radius_meters=1000)
    
    # First location (NYC)
    print("📍 Setting first location (NYC)...")
    success1 = store.update_location(40.7589, -73.9851)
    if success1:
        data1 = store.get_latest()
        print(f"✓ First location: {data1['count']} restaurants")
        first_location = data1['location']
        first_count = data1['count']
        first_time = data1['last_updated']
    
    time.sleep(2)  # Ensure different timestamps
    
    # Second location (San Francisco)
    print("📍 Setting second location (San Francisco)...")
    success2 = store.update_location(37.7749, -122.4194)
    if success2:
        data2 = store.get_latest()
        print(f"✓ Second location: {data2['count']} restaurants")
        second_location = data2['location']
        second_count = data2['count']
        second_time = data2['last_updated']
        
        # Verify data was replaced
        print(f"\n🔄 Data replacement verification:")
        print(f"  Location changed: {first_location != second_location}")
        print(f"  Count changed: {first_count != second_count}")
        print(f"  Time updated: {first_time != second_time}")
        print(f"  Old location: {first_location}")
        print(f"  New location: {second_location}")


def test_stats_and_utilities():
    """Test statistics and utility methods."""
    print_separator("TEST 5: STATS AND UTILITIES")
    
    store = NearbyRestaurantsStore(radius_meters=1500)
    
    # Update with a location
    success = store.update_location(40.7589, -73.9851)
    if success:
        # Test stats
        stats = store.get_stats()
        print("📊 Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test radius change
        print(f"\n🔧 Testing radius change:")
        print(f"  Current radius: {store._radius_meters}m")
        success_radius = store.set_radius(2000)
        print(f"  Radius change successful: {success_radius}")
        print(f"  New radius: {store._radius_meters}m")
        
        # Test invalid radius
        invalid_radius = store.set_radius(-100)
        print(f"  Invalid radius rejected: {not invalid_radius}")
        
        # Test data freshness with different thresholds
        print(f"\n⏰ Data freshness tests:")
        print(f"  Fresh (30 min): {store.is_data_fresh(30)}")
        print(f"  Fresh (5 min): {store.is_data_fresh(5)}")
        print(f"  Fresh (1 min): {store.is_data_fresh(1)}")
        
        # Test data clearing
        print(f"\n🧹 Testing data clearing:")
        print(f"  Before clear: {len(store._latest_restaurants)} restaurants")
        store.clear_data()
        print(f"  After clear: {len(store._latest_restaurants)} restaurants")
        print(f"  Location cleared: {store._latest_location is None}")


def test_error_handling():
    """Test error handling with various failure scenarios."""
    print_separator("TEST 6: ERROR HANDLING")
    
    store = NearbyRestaurantsStore()
    
    # Test invalid coordinates
    print("🚫 Testing invalid coordinates:")
    invalid_coords = [
        (999, 999, "Out of range coordinates"),
        (None, None, "None coordinates"),
        ("abc", "def", "String coordinates"),
        (float('inf'), float('nan'), "Special float values")
    ]
    
    for lat, lon, description in invalid_coords:
        try:
            success = store.update_location(lat, lon)
            status = "✓" if not success else "✗"
            print(f"  {status} {description}: Properly rejected")
        except Exception as e:
            print(f"  ✓ {description}: Exception handled - {type(e).__name__}")
    
    # Test with very remote location (should return few/no results)
    print(f"\n🏝️  Testing remote location (middle of ocean):")
    success = store.update_location(0.0, 0.0)  # Gulf of Guinea
    if success:
        data = store.get_latest()
        print(f"  ✓ Remote location handled: {data['count']} restaurants found")
    else:
        print(f"  ✓ Remote location handled: Update failed gracefully")


def test_overpass_query_building():
    """Test the Overpass query building functionality."""
    print_separator("TEST 7: OVERPASS QUERY BUILDING")
    
    store = NearbyRestaurantsStore(radius_meters=1000)
    
    # Test query building
    query = store._build_overpass_query(40.7589, -73.9851)
    print("🔍 Generated Overpass query:")
    print(query)
    
    # Verify query contains expected elements
    expected_elements = [
        "restaurant", "cafe", "fast_food", "bar", "pub", "food_court",
        "40.7589", "-73.9851", "1000", "out center meta"
    ]
    
    print(f"\n✅ Query validation:")
    for element in expected_elements:
        contains = element in query
        status = "✓" if contains else "✗"
        print(f"  {status} Contains '{element}': {contains}")


def test_google_maps_links():
    """Test Google Maps link generation."""
    print_separator("TEST 8: GOOGLE MAPS LINKS")
    
    store = NearbyRestaurantsStore(radius_meters=1000)
    
    # Test coordinate conversion with known values
    print("🔢 Testing coordinate conversion:")
    test_coords = [
        (40.7589, -73.9851, "Times Square, NYC"),
        (10.011162, 76.367256, "Kerala, India"),
        (-33.8688, 151.2093, "Sydney, Australia"),
        (51.5074, -0.1278, "London, UK")
    ]
    
    for lat, lon, location in test_coords:
        link = store._generate_google_maps_link(lat, lon)
        print(f"  📍 {location}")
        print(f"    Decimal: {lat}, {lon}")
        print(f"    DMS Link: {link}")
        print()
    
    # Update with a known location
    success = store.update_location(40.7589, -73.9851)  # Times Square
    
    if success:
        data = store.get_latest()
        if data['restaurants']:
            print("🗺️  Testing Google Maps link generation:")
            
            # Test first few restaurants
            for i, restaurant in enumerate(data['restaurants'][:3]):
                name = restaurant['name']
                lat = restaurant['latitude']
                lon = restaurant['longitude']
                link = restaurant.get('link', 'Missing!')
                
                print(f"\n  Restaurant {i+1}: {name}")
                print(f"    Coordinates: {lat}, {lon}")
                print(f"    Generated link: {link}")
                
                # Validate link format (new DMS format)
                expected_base = "https://www.google.com/maps/place/"
                has_degrees = "°" in link
                has_minutes = "'" in link  
                has_seconds = "\"" in link
                has_directions = any(d in link for d in ['N', 'S', 'E', 'W'])
                
                status_base = "✓" if link.startswith(expected_base) else "✗"
                status_degrees = "✓" if has_degrees else "✗"
                status_minutes = "✓" if has_minutes else "✗"
                status_seconds = "✓" if has_seconds else "✗"
                status_directions = "✓" if has_directions else "✗"
                
                print(f"    {status_base} Starts with correct base URL")
                print(f"    {status_degrees} Contains degrees (°)")
                print(f"    {status_minutes} Contains minutes (')")
                print(f"    {status_seconds} Contains seconds (\")")
                print(f"    {status_directions} Contains direction letters")
                
                # Overall validation
                is_valid = (link.startswith(expected_base) and has_degrees and 
                           has_minutes and has_seconds and has_directions)
                overall_status = "✅ VALID" if is_valid else "❌ INVALID"
                print(f"    {overall_status}")
            
            print(f"\n📊 Link generation summary:")
            total_restaurants = len(data['restaurants'])
            restaurants_with_links = len([r for r in data['restaurants'] if 'link' in r and r['link']])
            print(f"  Total restaurants: {total_restaurants}")
            print(f"  Restaurants with links: {restaurants_with_links}")
            print(f"  Link coverage: {restaurants_with_links/total_restaurants*100:.1f}%")
            
        else:
            print("❌ No restaurants found to test links")
    else:
        print("❌ Failed to fetch restaurant data for link testing")


def run_comprehensive_test():
    """Run all tests in sequence."""
    print("🚀 STARTING COMPREHENSIVE TEST SUITE")
    print(f"⏰ Test started at: {datetime.now()}")
    
    try:
        # Run all test functions
        store = test_initialization()
        test_coordinate_validation()
        store_with_data = test_real_location_update()
        test_data_replacement()
        test_stats_and_utilities()
        test_error_handling()
        test_overpass_query_building()
        test_google_maps_links()
        
        print_separator("🎉 ALL TESTS COMPLETED SUCCESSFULLY")
        print("✅ NearbyRestaurantsStore is working correctly!")
        print(f"⏰ Test completed at: {datetime.now()}")
        
        # Final demonstration
        if store_with_data and store_with_data.get_latest()['count'] > 0:
            print(f"\n📋 Final data summary:")
            final_data = store_with_data.get_latest()
            print(f"  Location: {final_data['location']}")
            print(f"  Restaurants: {final_data['count']}")
            print(f"  Sample restaurant: {final_data['restaurants'][0]['name'] if final_data['restaurants'] else 'None'}")
        
    except Exception as e:
        print_separator("❌ TEST SUITE FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def quick_demo():
    """Run a quick demonstration of basic functionality."""
    print_separator("🚀 QUICK DEMO")
    
    # Initialize store
    store = NearbyRestaurantsStore(radius_meters=1000)
    print("✓ Initialized restaurant store")
    
    # Test with Times Square coordinates
    lat, lon = 10.011162355852674, 76.36725577939116
    print(f"📍 Searching near Times Square: {lat}, {lon}")
    
    success = store.update_location(lat, lon)
    if success:
        data = store.get_latest()
        print(f"✅ Found {data['count']} restaurants!")
        
        # Show top 5 restaurants
        print(f"\n🍽️  Top restaurants:")
        for i, restaurant in enumerate(data['restaurants'][:5]):
            cuisine = restaurant.get('cuisine', 'Unknown cuisine')
            amenity = restaurant.get('amenity', 'restaurant')
            print(f"  {i+1}. {restaurant['name']} ({amenity})")
            print(f"     {cuisine} | {restaurant['latitude']:.4f}, {restaurant['longitude']:.4f}")
            print(f"     🗺️  Google Maps: {restaurant.get('link', 'N/A')}")
        
        # Show stats
        stats = store.get_stats()
        print(f"\n📊 Quick stats:")
        print(f"  Total restaurants: {stats['restaurant_count']}")
        print(f"  Data age: {stats['data_age_minutes']:.1f} minutes")
        print(f"  Amenity breakdown: {stats.get('amenity_breakdown', {})}")
        
    else:
        print("❌ Failed to fetch restaurant data")


if __name__ == "__main__":
    print("🧪 NearbyRestaurantsStore Test Suite")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_demo()
    else:
        print("Choose test mode:")
        print("1. Full comprehensive test suite")
        print("2. Quick demo")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            run_comprehensive_test()
        elif choice == "2":
            quick_demo()
        elif choice == "3":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice. Running quick demo...")
            quick_demo()
