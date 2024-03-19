def yes_no_question(question: str, empty_ok: bool = False):
    out = input(question + " [y/n]: ")
    return out == "y" or (empty_ok and out == "")
