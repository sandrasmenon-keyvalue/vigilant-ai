# Break Schedule Prediction Service

An LLM-based service that analyzes driver data to create personalized break schedules, predicting the next 5 break times to prevent drowsiness-related accidents.

## Features

- **LLM-powered break scheduling**: Uses Groq's GPT model via LangChain with structured output for intelligent break schedule creation
- **Next 5 breaks prediction**: Provides a complete schedule of the next 5 recommended breaks with timing and duration
- **Comprehensive input analysis**: Considers sleep debt, age, health conditions, driving duration, and time of day
- **Structured output**: Uses LangChain's built-in structured output for reliable JSON responses
- **Input validation**: Robust validation of all input parameters
- **Personalized scheduling**: Adapts break frequency and duration based on individual risk factors
- **Extensible design**: Easy to modify prompts and add new factors

## Installation

1. Install dependencies:
```bash
pip install -r ../../requirements.txt
```

2. Set up your Groq API key:
```bash
export GROQ_API_KEY="your-groq-api-key-here"
```

## Usage

### Basic Usage

```python
from datetime import datetime, timezone
from predict import (
    BreakPredictionService, 
    create_sleep_debt, 
    create_health_conditions
)

# Initialize the service
service = BreakPredictionService()

# Prepare input data
start_time = datetime.now(timezone.utc).replace(hour=8, minute=0)  # Started at 8 AM
sleep_debt = create_sleep_debt(duration_seconds=7200, sleep_quality=0.6)  # 2 hours sleep debt
age = 45
health_conditions = create_health_conditions(
    diabetes=False, 
    hypertension=True, 
    smoker=False
)

# Get break schedule
result = service.predict_next_breaks(start_time, sleep_debt, age, health_conditions)

print(f"Overall assessment: {result['overall_assessment']}")
print("Next 5 breaks:")
for i, break_info in enumerate(result['next_breaks'], 1):
    print(f"  Break {i}: {break_info['scheduled_time']} - {break_info['duration_minutes']}min - {break_info['reason']}")
```

### Input Parameters

#### `start_time` (datetime)
- UTC timestamp of when the driver started their current driving session
- Used to calculate total driving duration

#### `sleep_debt` (SleepDebt object)
- `duration`: Sleep deficit in seconds
- `sleep_quality`: Optional sleep quality score (0-1)

Create with: `create_sleep_debt(duration_seconds=7200, sleep_quality=0.7)`

#### `age` (int)
- Driver's age in years (16-120)
- Affects break recommendation thresholds

#### `health_conditions` (HealthConditions object)
- `diabetes`: Boolean
- `hypertension`: Boolean  
- `heart_disease`: Boolean
- `respiratory_condition`: Boolean
- `smoker`: Boolean

Create with: `create_health_conditions(diabetes=True, hypertension=False, ...)`

### Output Schema

```json
{
  "next_breaks": [
    {
      "scheduled_time": "2025-09-25T10:30:00-07:00",
      "duration_minutes": 20,
      "reason": "Regular 2-hour break"
    },
    {
      "scheduled_time": "2025-09-25T12:45:00-07:00", 
      "duration_minutes": 30,
      "reason": "Fatigue prevention break"
    },
    {
      "scheduled_time": "2025-09-25T15:15:00-07:00",
      "duration_minutes": 20,
      "reason": "Regular break"
    },
    {
      "scheduled_time": "2025-09-25T17:30:00-07:00",
      "duration_minutes": 45,
      "reason": "Extended rest for sleep debt"
    },
    {
      "scheduled_time": "2025-09-25T20:00:00-07:00",
      "duration_minutes": 30,
      "reason": "Late night alertness break"
    }
  ],
  "overall_assessment": "Driver shows moderate risk due to sleep debt and hypertension. Scheduled more frequent breaks with longer durations to maintain alertness.",
  "metadata": {
    "driving_duration_hours": 3.5,
    "sleep_debt_hours": 2.0,
    "age": 45,
    "health_conditions": {...},
    "timestamp": "2025-09-25T10:30:00Z",
    "user_timezone": "America/Los_Angeles"
  }
}
```

## Testing

Run the test example:
```bash
python test_example.py
```

This will test three scenarios and show the break schedules for:
1. High risk (long drive + high sleep debt + health conditions) - More frequent, longer breaks
2. Moderate risk (some sleep debt + age factor) - Regular intervals with moderate durations  
3. Low risk (young driver + minimal sleep debt) - Standard 2-hour intervals

## Customization

### Modifying the Prompt

Edit `prompts.yaml` to customize the LLM's break scheduling logic:

```yaml
break_schedule_prompt: |
  Your custom prompt here...
  Create 5 break times considering: {start_time}, {current_time}, {sleep_debt}, {age}, {health_conditions}
```

### Model Configuration

The service uses the "openai/gpt-oss-120b" model via Groq by default. You can modify this in the `BreakPredictionService` constructor:

```python
service = BreakPredictionService()
# The model is configured in the __init__ method
```

## Error Handling

The service includes comprehensive error handling:

- **Input validation**: Type checking and value range validation
- **API errors**: Handles Groq LLM request failures gracefully  
- **Parsing errors**: Validates LLM response format
- **Configuration errors**: Checks for missing prompts or Groq API keys

## Integration Example

```python
# Example integration in a driving app
def get_driver_break_schedule(driver_data):
    service = BreakPredictionService()
    
    try:
        result = service.predict_next_breaks(
            start_time=driver_data['session_start'],
            sleep_debt=driver_data['sleep_debt'],
            age=driver_data['age'],
            health_conditions=driver_data['health_conditions']
        )
        
        # Schedule notifications for upcoming breaks
        for break_info in result['next_breaks']:
            schedule_break_reminder(
                time=break_info['scheduled_time'],
                duration=break_info['duration_minutes'],
                reason=break_info['reason']
            )
            
        return result
        
    except Exception as e:
        # Handle errors gracefully
        log_error(f"Break schedule prediction failed: {e}")
        return create_default_break_schedule()

def create_default_break_schedule():
    """Fallback break schedule in case of system errors."""
    from datetime import datetime, timedelta
    now = datetime.now()
    return {
        "next_breaks": [
            {
                "scheduled_time": (now + timedelta(hours=2)).isoformat(),
                "duration_minutes": 20,
                "reason": "Standard break"
            }
            # ... 4 more breaks
        ],
        "overall_assessment": "Using default schedule due to system error"
    }
```

## Files

- `predict.py`: Main service implementation
- `prompts.yaml`: LLM prompts configuration
- `test_example.py`: Test scenarios and examples
- `README.md`: This documentation
