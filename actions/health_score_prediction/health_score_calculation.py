"""
Health Score Calculator
Simple function to calculate health score using formula: 0.6*dv + 0.4*hv
"""

def calculate_health_score(dv: float, hv: float) -> float:
    """
    Calculate health score using the formula: 0.6*dv + 0.4*hv
    
    Args:
        dv: Drowsiness score from vision model (0-1)
        hv: Health variability score from vitals model (0-1)
        
    Returns:
        Health score (0-1)
    """
    # If DV is 0 (no video stream), use HV-only scoring
    if dv == 0.0:
        # When no video is available, focus entirely on vitals
        # Scale HV score to 0-1 range for health score
        return hv
    
    # Normal case: combine DV and HV
    return 0.6 * dv + 0.4 * hv


def main():
    """Demo the simple health score calculation."""
    print("Health Score Calculator Demo")
    print("=" * 40)
    
    # Example usage
    dv_score = 0.3  # From vision model
    hv_score = 0.7  # From vitals model
    
    health_score = calculate_health_score(dv_score, hv_score)
    
    print(f"DV Score: {dv_score:.3f}")
    print(f"HV Score: {hv_score:.3f}")
    print(f"Health Score: {health_score:.3f}")
    print(f"Formula: 0.6*{dv_score:.3f} + 0.4*{hv_score:.3f} = {health_score:.3f}")


if __name__ == "__main__":
    main()
