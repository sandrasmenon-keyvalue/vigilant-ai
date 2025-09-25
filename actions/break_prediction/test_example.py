"""
Test example for the break prediction service.

This script demonstrates how to use the break prediction service with sample data.
Run this after installing the dependencies from requirements.txt.
"""

import os
from datetime import datetime, timezone
from predict import (
    BreakPredictionService, 
    create_sleep_debt, 
    create_health_conditions
)

def test_break_prediction():
    """Test the break prediction service with various scenarios."""
    
    # Check if API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY environment variable not set.")
        print("Please set your Groq API key to test the LLM integration.")
        return
    
    # Initialize the service
    print("Initializing Break Prediction Service...")
    service = BreakPredictionService()
    
    # Test scenarios with ISO string format (simulating client timezone)
    from datetime import timedelta
    import pytz
    
    # Simulate different timezones and times
    pst = pytz.timezone('America/Los_Angeles')
    est = pytz.timezone('America/New_York')
    
    test_scenarios = [
        {
            "name": "High Risk: Long drive + High sleep debt + Health conditions (Late night PST)",
            "start_time": (datetime.now(pst) - timedelta(hours=9)).isoformat(),  # Started 9 hours ago
            "sleep_debt": create_sleep_debt(duration_seconds=10800, sleep_quality=0.4),  # 3 hours, poor quality
            "age": 65,
            "health_conditions": create_health_conditions(diabetes=True, hypertension=True, smoker=True)
        },
        {
            "name": "Moderate Risk: Some sleep debt + Age factor (Early morning EST)",
            "start_time": (datetime.now(est) - timedelta(hours=3)).isoformat(),  # Started 3 hours ago
            "sleep_debt": create_sleep_debt(duration_seconds=5400, sleep_quality=0.7),  # 1.5 hours, good quality
            "age": 50,
            "health_conditions": create_health_conditions(hypertension=True)
        },
        {
            "name": "Low Risk: Young driver + Minimal sleep debt (Afternoon PST)",
            "start_time": (datetime.now(pst) - timedelta(hours=1)).isoformat(),  # Started 1 hour ago
            "sleep_debt": create_sleep_debt(duration_seconds=1800, sleep_quality=0.8),  # 30 minutes, good quality
            "age": 25,
            "health_conditions": create_health_conditions()  # No health conditions
        }
    ]
    
    # Run tests
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"Test Scenario {i}: {scenario['name']}")
        print(f"{'='*60}")
        
        try:
            result = service.predict_break_need(
                start_time=scenario["start_time"],
                sleep_debt=scenario["sleep_debt"],
                age=scenario["age"],
                health_conditions=scenario["health_conditions"]
            )
            
            print(f"Break Needed: {result['take_a_break']}")
            print(f"Duration: {result['duration']} seconds ({result['duration']/60:.1f} minutes)")
            print(f"Reason: {result['reason']}")
            print(f"Driving Duration: {result['metadata']['driving_duration_hours']:.1f} hours")
            print(f"Sleep Debt: {result['metadata']['sleep_debt_hours']:.1f} hours")
            
        except Exception as e:
            print(f"Error in scenario {i}: {e}")
    
    print(f"\n{'='*60}")
    print("Test completed!")

if __name__ == "__main__":
    test_break_prediction()
