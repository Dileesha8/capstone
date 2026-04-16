import numpy as np

def severity_score(probabilities, class_names):
    """
    Calculate severity score based on KL grading system for knee OA
    Typical grades: 0=Normal, 1=Doubtful, 2=Mild, 3=Moderate, 4=Severe
    
    Args:
        probabilities: array of class probabilities
        class_names: list of class names in order
    
    Returns:
        severity_score: 0-100 scale
    """
    # Define weights for each class (assuming KL grading)
    weights = {}
    
    # Map class names to weights (adjust based on your actual classes)
    for i, name in enumerate(class_names):
        name_lower = name.lower()
        if 'normal' in name_lower or 'healthy' in name_lower:
            weights[i] = 0
        elif 'doubtful' in name_lower or 'minimal' in name_lower:
            weights[i] = 25
        elif 'mild' in name_lower:
            weights[i] = 50
        elif 'moderate' in name_lower:
            weights[i] = 75
        elif 'severe' in name_lower or 'advanced' in name_lower:
            weights[i] = 100
        else:
            # Default fallback
            weights[i] = (i / (len(class_names) - 1)) * 100
    
    # Calculate weighted score
    score = sum(probabilities[i] * weights[i] for i in range(len(probabilities)))
    
    return round(score, 2)

def forecast_risk(severity_score):
    """
    Forecast risk based on severity score
    """
    if severity_score < 20:
        return "Low Risk (10-15%)", "Minimal joint space narrowing, regular monitoring recommended"
    elif severity_score < 40:
        return "Mild Risk (25-30%)", "Early OA signs, lifestyle modifications advised"
    elif severity_score < 60:
        return "Moderate Risk (50-60%)", "Definite OA, consider treatment options"
    elif severity_score < 80:
        return "High Risk (70-80%)", "Advanced OA, medical intervention recommended"
    else:
        return "Severe Risk (90%+)", "Severe OA, surgical consultation recommended"