def severity_score(probabilities):
    """
    probabilities: model output like [0.2, 0.5, 0.3]
    """
    weights = {
        "mild": 0.3,
        "moderate": 0.6,
        "severe": 0.9
    }

    score = (
        probabilities[0] * weights["mild"] +
        probabilities[1] * weights["moderate"] +
        probabilities[2] * weights["severe"]
    )

    return round(score * 100, 2)


