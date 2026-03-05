
def print_results(acc, scores_test, model):

    print("Accuracy:", acc)
    print("Scores:", scores_test[:10])
    print(f"Thresholds used → t1: {model.t1}, t2: {model.t2}")