"""
Test timezone functionality for break prediction.
"""

import pytz
from datetime import datetime, timedelta
from predict import BreakPredictionService, create_sleep_debt, create_health_conditions

def test_timezone_parsing():
    """Test timezone parsing and time calculations."""
    
    service = BreakPredictionService()
    
    # Test different timezone formats
    test_times = [
        # PST timezone
        "2025-09-25T08:00:00-07:00",  # 8 AM PST
        # EST timezone  
        "2025-09-25T11:00:00-04:00",  # 11 AM EST
        # UTC timezone
        "2025-09-25T15:00:00+00:00",  # 3 PM UTC
    ]
    
    print("🕒 Testing Timezone Parsing:")
    for i, time_str in enumerate(test_times, 1):
        try:
            parsed_time = service._parse_start_time(time_str)
            current_info = service._get_current_time_info(parsed_time)
            duration = service._calculate_driving_duration(parsed_time)
            
            print(f"\nTest {i}: {time_str}")
            print(f"  ✅ Parsed: {parsed_time}")
            print(f"  ✅ Current time: {current_info['current_time']}")
            print(f"  ✅ Driving duration: {duration:.2f} hours")
            print(f"  ✅ Timezone: {current_info['timezone_name']}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Test with actual prediction (no API call)
    print(f"\n{'='*50}")
    print("🚗 Testing Full Prediction with Timezone:")
    
    # Simulate a 2-hour drive starting in PST
    pst = pytz.timezone('America/Los_Angeles')
    start_time_pst = (datetime.now(pst) - timedelta(hours=2)).isoformat()
    
    print(f"Start time (PST): {start_time_pst}")
    
    try:
        # This will work without API key for parsing/validation
        service._validate_inputs(
            start_time_pst,
            create_sleep_debt(3600, 0.7),  # 1 hour sleep debt
            35,
            create_health_conditions()
        )
        
        parsed = service._parse_start_time(start_time_pst)
        duration = service._calculate_driving_duration(parsed)
        current_info = service._get_current_time_info(parsed)
        
        print(f"✅ Validation passed")
        print(f"✅ Driving duration: {duration:.2f} hours")
        print(f"✅ Current time in user's timezone: {current_info['current_time']}")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")

if __name__ == "__main__":
    test_timezone_parsing()
