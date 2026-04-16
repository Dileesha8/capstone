def forecast_risk(score):
    if score < 40:
        return "Low Risk (15%)"
    elif score < 70:
        return "Moderate Risk (45%)"
    else:
        return "High Risk (80%)"
