def assign_severity(opacity_ratio):
    if opacity_ratio < 0.2:
        return "mild"
    elif opacity_ratio < 0.5:
        return "moderate"
    else:
        return "severe"
