# utils/formatting.py

def color_code_index(index_name, value):
    if index_name == "Flesch Reading Ease":
        if value >= 90:
            color = "green"
        elif 60 <= value < 90:
            color = "lightgreen"
        elif 30 <= value < 60:
            color = "orange"
        else:
            color = "red"
    elif index_name == "Flesch-Kincaid Grade Level":
        if value <= 5:
            color = "green"
        elif 6 <= value <= 10:
            color = "lightgreen"
        elif 11 <= value <= 15:
            color = "orange"
        else:
            color = "red"
    elif index_name in ["Gunning Fog Index", "SMOG Index"]:
        if value <= 6:
            color = "green"
        elif 7 <= value <= 12:
            color = "lightgreen"
        elif 13 <= value <= 17:
            color = "orange"
        else:
            color = "red"
    else:
        color = "black"
    return f"<span style='color: {color};'>{value:.2f}</span>"