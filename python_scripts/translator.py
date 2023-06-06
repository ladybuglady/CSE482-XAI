import numpy as np

def translate(shap: list, features: dict, sense=5) -> list:

    sense = sense
    time = 100
    
    sentences_lead_1 = translate_lead(shap[0], features, sense, time, "I")
    sentences_lead_2 = translate_lead(shap[1], features, sense, time, "II")

    return sentences_lead_1 + sentences_lead_2

def translate_lead(shap: list, features: dict, sense: int, time: int, lead: str) -> list:

    sentences = []

    i = 0
    while i < (len(shap) - time - 1):
        if sum(1 for x in shap[i:i + time] if x > 0) >= sense:
            indexes = []
            for j in range(i, i + time):
                if shap[j] > 0:
                    indexes.append(j)
            
            for k, v in features.items():
                if k in ("R1", "R2"):
                    continue

                end = False

                for p in v:
                    if sum(1 for x in indexes if p[0] <= x <= p[1]) / sense >= 0.5:
                        for index in indexes:
                            if p[0] <= index <= p[1]:
                                if k in ("P", "T"):
                                    sentences.append("The model has detected a " + k + "-wave abnormality at " + str(index) + " ms in lead " + lead + ".")
                                elif k == "QRS":
                                    sentences.append("The model has detected a " + k + "-complex abnormality at " + str(index) + " ms in lead " + lead + ".")
                                else:
                                    sentences.append("The model has detected a " + k + "-segment abnormality at " + str(index) + " ms in lead " + lead + ".")
                                end = True
                                break
                        if end:
                            break
                if end:
                    break

            i = i + time
        else:
            i += 1
    
    return sentences