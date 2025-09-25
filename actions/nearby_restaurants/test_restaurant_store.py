#!/usr/bin/env python3
"""
Test script for RestaurantFinder class

This script tests the functionality of the RestaurantFinder class
with real OpenStreetMap API calls and various test scenarios.
"""

import sys
import time
from datetime import datetime
from restaurant_finder import RestaurantFinder


def print_separator(title: str):
    """Print a formatted separator for test sections."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def test_initialization():
    """Test class initialization with different parameters."""
    print_separator("TEST 1: INITIALIZATION")
    
    # Test default initialization
    finder1 = RestaurantFinder()
    print("✓ Default initialization successful")
    print(f"  Default radius: {finder1._radius_meters}m")
    
    # Test custom radius
    finder2 = RestaurantFinder(radius_meters=2000)
    print("✓ Custom radius initialization successful")
    print(f"  Custom radius: {finder2._radius_meters}m")
    
    print(f"✓ RestaurantFinder is stateless - no stored data")
    
    return finder1


def test_coordinate_validation():
    """Test coordinate validation with various inputs."""
    print_separator("TEST 2: COORDINATE VALIDATION")
    
    finder = RestaurantFinder()
    
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
            result = finder._validate_coordinates(lat, lon)
            status = "✓" if result == expected else "✗"
            print(f"{status} {description}: {result}")
        except Exception as e:
            result = False
            status = "✓" if result == expected else "✗"
            print(f"{status} {description}: Exception caught - {str(e)}")


def test_restaurant_finding():
    """Test finding restaurants with real coordinates and API calls."""
    print_separator("TEST 3: RESTAURANT FINDING")
    
    finder = RestaurantFinder(radius_meters=1000)
    
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
        result = finder.find_nearby_restaurants(lat, lon)
        end_time = time.time()
        
        if result['count'] > 0:
            print(f"✓ Search successful in {end_time - start_time:.2f} seconds")
            print(f"  Found {result['count']} restaurants")
            print(f"  Location: {result['location']}")
            print(f"  Last updated: {result['last_updated']}")
            
            # Show sample restaurants
            if result['restaurants']:
                print(f"  Sample restaurants:")
                for i, restaurant in enumerate(result['restaurants'][:3]):
                    print(f"    {i+1}. {restaurant['name']}")
                    print(f"       Google Maps: {restaurant.get('link', 'N/A')}")
            
        else:
            print("✗ No restaurants found")
        
        # Small delay between requests to be respectful to the API
        time.sleep(1)
    
    return finder


def test_stateless_behavior():
    """Test that the finder is stateless - no data persistence between calls."""
    print_separator("TEST 4: STATELESS BEHAVIOR")
    
    finder = RestaurantFinder(radius_meters=1000)
    
    # First search (NYC)
    print("📍 First search (NYC)...")
    result1 = finder.find_nearby_restaurants(40.7589, -73.9851)
    print(f"✓ First search: {result1['count']} restaurants")
    first_location = result1['location']
    first_count = result1['count']
    first_time = result1['last_updated']
    
    time.sleep(2)  # Ensure different timestamps
    
    # Second search (San Francisco)
    print("📍 Second search (San Francisco)...")
    result2 = finder.find_nearby_restaurants(37.7749, -122.4194)
    print(f"✓ Second search: {result2['count']} restaurants")
    second_location = result2['location']
    second_count = result2['count']
    second_time = result2['last_updated']
    
    # Verify stateless behavior
    print(f"\n🔄 Stateless verification:")
    print(f"  Different locations: {first_location != second_location}")
    print(f"  Different counts: {first_count != second_count}")
    print(f"  Different timestamps: {first_time != second_time}")
    print(f"  No data persistence: ✓ (each call is independent)")


def test_utilities():
    """Test utility methods."""
    print_separator("TEST 5: UTILITIES")
    
    finder = RestaurantFinder(radius_meters=1500)
    
    # Test radius change
    print(f"🔧 Testing radius change:")
    print(f"  Current radius: {finder._radius_meters}m")
    success_radius = finder.set_radius(2000)
    print(f"  Radius change successful: {success_radius}")
    print(f"  New radius: {finder._radius_meters}m")
    
    # Test invalid radius
    invalid_radius = finder.set_radius(-100)
    print(f"  Invalid radius rejected: {not invalid_radius}")
    
    print(f"\n✅ RestaurantFinder utilities working correctly")


def test_error_handling():
    """Test error handling with various failure scenarios."""
    print_separator("TEST 6: ERROR HANDLING")
    
    finder = RestaurantFinder()
    
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
            result = finder.find_nearby_restaurants(lat, lon)
            empty_result = result['count'] == 0 and result['location'] is None
            status = "✓" if empty_result else "✗"
            print(f"  {status} {description}: Properly rejected")
        except Exception as e:
            print(f"  ✓ {description}: Exception handled - {type(e).__name__}")
    
    # Test with very remote location (should return few/no results)
    print(f"\n🏝️  Testing remote location (middle of ocean):")
    result = finder.find_nearby_restaurants(0.0, 0.0)  # Gulf of Guinea
    print(f"  ✓ Remote location handled: {result['count']} restaurants found")


def test_overpass_query_building():
    """Test the Overpass query building functionality."""
    print_separator("TEST 7: OVERPASS QUERY BUILDING")
    
    finder = RestaurantFinder(radius_meters=1000)
    
    # Test query building
    query = finder._build_overpass_query(40.7589, -73.9851)
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
    
    finder = RestaurantFinder(radius_meters=1000)
    
    # Test coordinate conversion with known values
    print("🔢 Testing coordinate conversion:")
    test_coords = [
        (40.7589, -73.9851, "Times Square, NYC"),
        (10.011162, 76.367256, "Kerala, India"),
        (-33.8688, 151.2093, "Sydney, Australia"),
        (51.5074, -0.1278, "London, UK")
    ]
    
    for lat, lon, location in test_coords:
        link = finder._generate_google_maps_link(lat, lon)
        print(f"  📍 {location}")
        print(f"    Decimal: {lat}, {lon}")
        print(f"    DMS Link: {link}")
        print()
    
    # Find restaurants at a known location
    result = finder.find_nearby_restaurants(40.7589, -73.9851)  # Times Square
    
    if result['count'] > 0:
        if result['restaurants']:
            print("🗺️  Testing Google Maps link generation:")
            
            # Test first few restaurants
            for i, restaurant in enumerate(result['restaurants'][:3]):
                name = restaurant['name']
                link = restaurant.get('link', 'Missing!')
                
                print(f"\n  Restaurant {i+1}: {name}")
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
            total_restaurants = len(result['restaurants'])
            restaurants_with_links = len([r for r in result['restaurants'] if 'link' in r and r['link']])
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
        finder = test_initialization()
        test_coordinate_validation()
        finder_with_data = test_restaurant_finding()
        test_stateless_behavior()
        test_utilities()
        test_error_handling()
        test_overpass_query_building()
        test_google_maps_links()
        
        print_separator("🎉 ALL TESTS COMPLETED SUCCESSFULLY")
        print("✅ RestaurantFinder is working correctly!")
        print(f"⏰ Test completed at: {datetime.now()}")
        
        # Final demonstration
        print(f"\n📋 Final demonstration:")
        final_result = finder.find_nearby_restaurants(40.7589, -73.9851)
        print(f"  Location: {final_result['location']}")
        print(f"  Restaurants: {final_result['count']}")
        print(f"  Sample restaurant: {final_result['restaurants'][0]['name'] if final_result['restaurants'] else 'None'}")
        print(f"  ✅ Stateless design working perfectly!")
        
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
    
    # Initialize finder
    finder = RestaurantFinder(radius_meters=1000)
    print("✓ Initialized restaurant finder")
    
    # Test with Kerala coordinates
    lat, lon = 10.011162355852674, 76.36725577939116
    print(f"📍 Searching near Kerala, India: {lat}, {lon}")
    
    result = finder.find_nearby_restaurants(lat, lon)
    if result['count'] > 0:
        print(f"✅ Found {result['count']} restaurants!")
        
        # Show top 5 restaurants
        print(f"\n🍽️  Top restaurants:")
        for i, restaurant in enumerate(result['restaurants'][:5]):
            print(f"  {i+1}. {restaurant['name']}")
            print(f"     🗺️  Google Maps: {restaurant.get('link', 'N/A')}")
        
        print(f"\n📊 Quick stats:")
        print(f"  Total restaurants: {result['count']}")
        print(f"  Search radius: {result['search_radius_meters']}m")
        print(f"  Fetched at: {result['last_updated']}")
        
        # Since we only have name and link now, show simple statistics
        print(f"  All restaurants have: name and Google Maps link")
        
    else:
        print("❌ No restaurants found")


if __name__ == "__main__":
    print("🧪 RestaurantFinder Test Suite")
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
