"""
Quick test to verify structured output implementation.
"""

import os
from datetime import datetime, timezone
from predict import (
    BreakPredictionService, 
    create_sleep_debt, 
    create_health_conditions,
    BreakPredictionResponse
)

def test_structured_output():
    """Test that structured output returns proper Pydantic objects."""
    
    # Check if we can instantiate the service
    try:
        service = BreakPredictionService()
        print("✅ Service initialized successfully")
        print(f"✅ LLM configured with structured output: {type(service.llm)}")
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return
    
    # Test Pydantic model validation
    try:
        # Valid response
        valid_response = BreakPredictionResponse(
            take_a_break=True,
            reason="Test reason",
            duration=1800.0
        )
        print(f"✅ Valid response created: {valid_response}")
        
        # Invalid response (should fail validation)
        try:
            invalid_response = BreakPredictionResponse(
                take_a_break=False,  # No break needed
                reason="Test reason", 
                duration=1800.0  # But duration > 0 (should fail)
            )
            print("❌ Validation should have failed but didn't")
        except Exception as validation_error:
            print(f"✅ Validation correctly caught error: {validation_error}")
            
    except Exception as e:
        print(f"❌ Pydantic model test failed: {e}")
    
    # Test with API key (only if available)
    if os.getenv("GROQ_API_KEY"):
        print("\n🔄 Testing with actual API call...")
        try:
            start_time = datetime.now(timezone.utc).replace(hour=8, minute=0)
            sleep_debt = create_sleep_debt(duration_seconds=3600, sleep_quality=0.7)
            health_conditions = create_health_conditions(hypertension=True)
            
            result = service.predict_break_need(start_time, sleep_debt, 45, health_conditions)
            
            print(f"✅ API call successful")
            print(f"✅ Response type: {type(result)}")
            print(f"✅ Break needed: {result['take_a_break']}")
            print(f"✅ Duration: {result['duration']} seconds")
            
        except Exception as e:
            print(f"❌ API call failed: {e}")
    else:
        print("\n⚠️  GROQ_API_KEY not set, skipping API test")
    
    print("\n✅ Structured output test completed!")

if __name__ == "__main__":
    test_structured_output()
