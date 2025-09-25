"""
Alert Prediction Module
Simple function to create Alert objects with warning type and reason
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Alert:
    """Alert object containing type and reason."""
    type: str
    reason: str


def create_alert(reason: str) -> Alert:
    """
    Create an Alert object with type "warning" and the provided reason.
    
    Args:
        reason: The reason for the alert
        
    Returns:
        Alert object with type "warning" and the specified reason
    """
    return Alert(type="warning", reason=reason)


def main():
    """Demo the alert creation function."""
    print("Alert Prediction Demo")
    print("=" * 30)
    
    # Example usage
    alert = create_alert("Health score is above threshold")
    
    print(f"Alert Type: {alert.type}")
    print(f"Alert Reason: {alert.reason}")


if __name__ == "__main__":
    main()
