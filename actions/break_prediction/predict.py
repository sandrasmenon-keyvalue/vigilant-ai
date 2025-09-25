"""
Break Prediction Service using LLM-based analysis.

This module analyzes driver data including sleep debt, health conditions, and driving duration
to predict if a driver should take a break to prevent drowsiness-related accidents.
"""

import json
import yaml
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import pytz
from dateutil import parser

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field, field_validator
from typing import List

load_dotenv()

class BreakPredictionResponse(BaseModel):
    """Pydantic model for break prediction response validation."""
    break_times: List[str] = Field(description="List of exactly 5 ISO timestamp strings for when breaks should be taken")
    
    @field_validator('break_times')
    @classmethod
    def validate_break_times(cls, v):
        """Ensure we have exactly 5 break times as ISO strings."""
        if len(v) != 5:
            raise ValueError("Must provide exactly 5 break times")
        
        for i, break_time in enumerate(v):
            if not isinstance(break_time, str):
                raise ValueError(f"Break time {i+1} must be a string (ISO timestamp)")
        return v


@dataclass
class SleepDebt:
    """Sleep debt information."""
    duration: int  # Duration in seconds
    sleep_quality: Optional[float] = None  # Quality score 0-1 (optional)


@dataclass
class HealthConditions:
    """Health conditions that may affect driver alertness."""
    diabetes: bool = False
    hypertension: bool = False
    heart_disease: bool = False
    respiratory_condition: bool = False
    smoker: bool = False


class BreakPredictionService:
    """LLM-based break prediction service for driver drowsiness prevention."""
    
    def __init__(self, prompts_path: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the break prediction service.
        
        Args:
            prompts_path: Path to prompts.yaml file
            api_key: Groq API key (if not set via environment)
        """
        self.prompts_path = prompts_path or self._get_default_prompts_path()
        self.prompts = self._load_prompts()
        
        # Initialize LangChain with Groq model and structured output
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.1,  # Low temperature for consistent responses
            groq_api_key=api_key or os.getenv("GROQ_API_KEY")
        ).with_structured_output(BreakPredictionResponse)
    
    def _get_default_prompts_path(self) -> str:
        """Get default path to prompts.yaml file."""
        current_dir = Path(__file__).parent
        return str(current_dir / "prompts.yaml")
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from YAML file."""
        try:
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
            return prompts
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompts file not found at {self.prompts_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
    
    def _parse_start_time(self, start_time_str: str) -> datetime:
        """Parse ISO string to datetime object, preserving timezone."""
        try:
            # Parse ISO string with timezone info
            parsed_time = parser.isoparse(start_time_str)
            return parsed_time
        except Exception as e:
            raise ValueError(f"Invalid start_time format. Expected ISO string with timezone: {e}")
    
    def _calculate_driving_duration(self, start_time: datetime) -> float:
        """Calculate driving duration in hours from start time."""
        # Get current time in the same timezone as start_time
        user_timezone = start_time.tzinfo
        now = datetime.now(user_timezone)
        
        duration = (now - start_time).total_seconds() / 3600  # Convert to hours
        return max(0, duration)  # Ensure non-negative
    
    def _get_current_time_info(self, start_time: datetime) -> Dict[str, str]:
        """Get current time information in user's timezone."""
        user_timezone = start_time.tzinfo
        now = datetime.now(user_timezone)
        
        return {
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "current_hour": now.hour,
            "timezone_name": str(user_timezone)
        }
    
    def _format_sleep_debt(self, sleep_debt: SleepDebt) -> str:
        """Format sleep debt information for the prompt."""
        hours = sleep_debt.duration / 3600
        result = f"Duration: {hours:.1f} hours ({sleep_debt.duration} seconds)"
        if sleep_debt.sleep_quality is not None:
            result += f", Quality: {sleep_debt.sleep_quality:.2f}/1.0"
        return result
    
    def _format_health_conditions(self, health_conditions: HealthConditions) -> str:
        """Format health conditions for the prompt."""
        conditions = {
            "diabetes": health_conditions.diabetes,
            "hypertension": health_conditions.hypertension,
            "heart_disease": health_conditions.heart_disease,
            "respiratory_condition": health_conditions.respiratory_condition,
            "smoker": health_conditions.smoker
        }
        return json.dumps(conditions)
    
    def _format_previous_break_times(self, previous_break_times: Optional[List[str]]) -> str:
        """Format previous break times for the prompt."""
        if not previous_break_times:
            return "None (first prediction)"
        return json.dumps(previous_break_times)
    
    def _validate_inputs(self, start_time, sleep_debt: SleepDebt, 
                        age: int, health_conditions: HealthConditions,
                        previous_break_times: Optional[List[str]] = None) -> None:
        """Validate input parameters."""
        if not isinstance(start_time, (datetime, str)):
            raise TypeError("start_time must be a datetime object or ISO string")
        
        if not isinstance(sleep_debt, SleepDebt):
            raise TypeError("sleep_debt must be a SleepDebt object")
        
        if sleep_debt.duration < 0:
            raise ValueError("sleep_debt.duration must be non-negative")
        
        if not isinstance(age, int) or age < 16 or age > 120:
            raise ValueError("age must be an integer between 16 and 120")
        
        if not isinstance(health_conditions, HealthConditions):
            raise TypeError("health_conditions must be a HealthConditions object")
        
        if previous_break_times is not None:
            if not isinstance(previous_break_times, list):
                raise TypeError("previous_break_times must be a list or None")
            for i, break_time in enumerate(previous_break_times):
                if not isinstance(break_time, str):
                    raise TypeError(f"previous_break_times[{i}] must be a string (ISO timestamp)")
    
    def predict_next_breaks(self, start_time, sleep_debt: SleepDebt, 
                           age: int, health_conditions: HealthConditions,
                           previous_break_times: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Predict the next 5 break times for a driver based on their current state and previous predictions.
        
        Args:
            start_time: When the driver started driving (datetime object or ISO string with timezone)
            sleep_debt: Sleep debt information
            age: Driver's age in years
            health_conditions: Driver's health conditions
            previous_break_times: List of previously predicted break times (ISO strings). Can be empty or None.
            
        Returns:
            Dictionary with next 5 break times
            
        Raises:
            ValueError: If input validation fails
            TypeError: If input types are incorrect
            RuntimeError: If LLM request fails
        """
        # Validate inputs
        self._validate_inputs(start_time, sleep_debt, age, health_conditions, previous_break_times)
        
        # Parse start_time if it's a string
        if isinstance(start_time, str):
            start_time = self._parse_start_time(start_time)
        
        # Calculate additional context
        driving_duration = self._calculate_driving_duration(start_time)
        current_time_info = self._get_current_time_info(start_time)
        
        # Format data for prompt
        formatted_sleep_debt = self._format_sleep_debt(sleep_debt)
        formatted_health = self._format_health_conditions(health_conditions)
        formatted_previous_times = self._format_previous_break_times(previous_break_times)
        
        # Prepare the prompt
        prompt_template = self.prompts.get('break_times_prompt', '')
        if not prompt_template:
            raise ValueError("break_times_prompt not found in prompts file")
        
        formatted_prompt = prompt_template.format(
            start_time=start_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            current_time=current_time_info["current_time"],
            driving_duration_hours=f"{driving_duration:.1f}",
            sleep_debt=formatted_sleep_debt,
            age=age,
            health_conditions=formatted_health,
            previous_break_times=formatted_previous_times
        )
        
        try:
            # Make LLM request with structured output
            message = HumanMessage(content=formatted_prompt)
            parsed_response = self.llm.invoke([message])
            
            # Add metadata
            result = {
                "break_times": parsed_response.break_times,
                "metadata": {
                    "driving_duration_hours": driving_duration,
                    "sleep_debt_hours": sleep_debt.duration / 3600,
                    "age": age,
                    "health_conditions": health_conditions.__dict__,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "user_timezone": str(start_time.tzinfo)
                }
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to get break schedule prediction: {str(e)}")


def create_sleep_debt(duration_seconds: int, sleep_quality: Optional[float] = None) -> SleepDebt:
    """Helper function to create SleepDebt object."""
    return SleepDebt(duration=duration_seconds, sleep_quality=sleep_quality)


def create_health_conditions(diabetes: bool = False, hypertension: bool = False,
                           heart_disease: bool = False, respiratory_condition: bool = False,
                           smoker: bool = False) -> HealthConditions:
    """Helper function to create HealthConditions object."""
    return HealthConditions(
        diabetes=diabetes,
        hypertension=hypertension,
        heart_disease=heart_disease,
        respiratory_condition=respiratory_condition,
        smoker=smoker
    )


# Example usage
if __name__ == "__main__":
    # Example usage of the break prediction service
    service = BreakPredictionService()
    
    # Sample data
    start_time = datetime.now(timezone.utc).replace(hour=8, minute=0)  # Started driving at 8 AM
    sleep_debt = create_sleep_debt(duration_seconds=7200, sleep_quality=0.6)  # 2 hours sleep debt
    age = 45
    health_conditions = create_health_conditions(diabetes=False, hypertension=True, smoker=False)
    
    try:
        # First prediction (no previous break times)
        result = service.predict_next_breaks(start_time, sleep_debt, age, health_conditions)
        print("First Break Times Prediction:")
        print(json.dumps(result, indent=2))
        
        # Example of subsequent prediction with previous break times
        previous_times = result["break_times"][:3]  # Simulate that some breaks have passed
        result2 = service.predict_next_breaks(start_time, sleep_debt, age, health_conditions, previous_times)
        print("\nSecond Break Times Prediction (with previous times):")
        print(json.dumps(result2, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
