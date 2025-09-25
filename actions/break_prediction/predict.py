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

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field, validator

load_dotenv()

class BreakPredictionResponse(BaseModel):
    """Pydantic model for break prediction response validation."""
    take_a_break: bool = Field(description="Whether the user should take a break or not")
    reason: str = Field(description="Reason for the above response")
    duration: float = Field(description="Seconds to take a break, 0 if no need to take a break")
    
    @validator('duration')
    def validate_duration(cls, v, values):
        """Ensure duration is 0 when no break is needed."""
        if not values.get('take_a_break', False) and v > 0:
            raise ValueError("Duration should be 0 when no break is needed")
        if values.get('take_a_break', False) and v <= 0:
            raise ValueError("Duration should be > 0 when break is needed")
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
    
    def _calculate_driving_duration(self, start_time: datetime) -> float:
        """Calculate driving duration in hours from start time."""
        now = datetime.now(timezone.utc)
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        duration = (now - start_time).total_seconds() / 3600  # Convert to hours
        return max(0, duration)  # Ensure non-negative
    
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
    
    def _validate_inputs(self, start_time: datetime, sleep_debt: SleepDebt, 
                        age: int, health_conditions: HealthConditions) -> None:
        """Validate input parameters."""
        if not isinstance(start_time, datetime):
            raise TypeError("start_time must be a datetime object")
        
        if not isinstance(sleep_debt, SleepDebt):
            raise TypeError("sleep_debt must be a SleepDebt object")
        
        if sleep_debt.duration < 0:
            raise ValueError("sleep_debt.duration must be non-negative")
        
        if not isinstance(age, int) or age < 16 or age > 120:
            raise ValueError("age must be an integer between 16 and 120")
        
        if not isinstance(health_conditions, HealthConditions):
            raise TypeError("health_conditions must be a HealthConditions object")
    
    def predict_break_need(self, start_time: datetime, sleep_debt: SleepDebt, 
                          age: int, health_conditions: HealthConditions) -> Dict[str, Any]:
        """
        Predict if a driver needs to take a break based on their current state.
        
        Args:
            start_time: When the driver started driving (UTC timestamp)
            sleep_debt: Sleep debt information
            age: Driver's age in years
            health_conditions: Driver's health conditions
            
        Returns:
            Dictionary with break prediction results
            
        Raises:
            ValueError: If input validation fails
            TypeError: If input types are incorrect
            RuntimeError: If LLM request fails
        """
        # Validate inputs
        self._validate_inputs(start_time, sleep_debt, age, health_conditions)
        
        # Calculate additional context
        driving_duration = self._calculate_driving_duration(start_time)
        
        # Format data for prompt
        formatted_sleep_debt = self._format_sleep_debt(sleep_debt)
        formatted_health = self._format_health_conditions(health_conditions)
        
        # Prepare the prompt
        prompt_template = self.prompts.get('break_prediction_prompt', '')
        if not prompt_template:
            raise ValueError("break_prediction_prompt not found in prompts file")
        
        # Add driving duration context to the prompt
        enhanced_start_time = f"{start_time.isoformat()} (driving for {driving_duration:.1f} hours)"
        
        formatted_prompt = prompt_template.format(
            start_time=enhanced_start_time,
            sleep_debt=formatted_sleep_debt,
            age=age,
            health_conditions=formatted_health
        )
        
        try:
            # Make LLM request with structured output
            message = HumanMessage(content=formatted_prompt)
            parsed_response = self.llm.invoke([message])
            
            # Add metadata
            result = {
                "take_a_break": parsed_response.take_a_break,
                "reason": parsed_response.reason,
                "duration": parsed_response.duration,
                "metadata": {
                    "driving_duration_hours": driving_duration,
                    "sleep_debt_hours": sleep_debt.duration / 3600,
                    "age": age,
                    "health_conditions": health_conditions.__dict__,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to get break prediction: {str(e)}")


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
        result = service.predict_break_need(start_time, sleep_debt, age, health_conditions)
        print("Break Prediction Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
