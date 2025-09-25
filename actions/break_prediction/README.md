# Break Prediction Service

An LLM-based service that analyzes driver data to predict when they should take breaks to prevent drowsiness-related accidents.

## Features

- **LLM-powered analysis**: Uses Groq's GPT model via LangChain with structured output for intelligent break recommendations
- **Comprehensive input analysis**: Considers sleep debt, age, health conditions, and driving duration
- **Structured output**: Uses LangChain's built-in structured output for reliable JSON responses
- **Input validation**: Robust validation of all input parameters
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

# Get prediction
result = service.predict_break_need(start_time, sleep_debt, age, health_conditions)

print(f"Break needed: {result['take_a_break']}")
print(f"Reason: {result['reason']}")
print(f"Duration: {result['duration']} seconds")
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
  "take_a_break": true,
  "reason": "Driver has significant sleep debt (2.0 hours) combined with hypertension...",
  "duration": 1800.0,
  "metadata": {
    "driving_duration_hours": 3.5,
    "sleep_debt_hours": 2.0,
    "age": 45,
    "health_conditions": {...},
    "timestamp": "2025-09-25T10:30:00Z"
  }
}
```

## Testing

Run the test example:
```bash
python test_example.py
```

This will test three scenarios:
1. High risk (long drive + high sleep debt + health conditions)
2. Moderate risk (some sleep debt + age factor)
3. Low risk (young driver + minimal sleep debt)

## Customization

### Modifying the Prompt

Edit `prompts.yaml` to customize the LLM's decision-making logic:

```yaml
break_prediction_prompt: |
  Your custom prompt here...
  Consider these factors: {start_time}, {sleep_debt}, {age}, {health_conditions}
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
def check_driver_status(driver_data):
    service = BreakPredictionService()
    
    try:
        result = service.predict_break_need(
            start_time=driver_data['session_start'],
            sleep_debt=driver_data['sleep_debt'],
            age=driver_data['age'],
            health_conditions=driver_data['health_conditions']
        )
        
        if result['take_a_break']:
            # Trigger break notification
            notify_driver(result['reason'], result['duration'])
            
        return result
        
    except Exception as e:
        # Handle errors gracefully
        log_error(f"Break prediction failed: {e}")
        return {"take_a_break": True, "reason": "System error - taking precautionary break", "duration": 900}
```

## Files

- `predict.py`: Main service implementation
- `prompts.yaml`: LLM prompts configuration
- `test_example.py`: Test scenarios and examples
- `README.md`: This documentation
