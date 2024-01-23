def yes_no_question(question: str):
    out = input(question + " [y/n]: ")
    return out == "y"
