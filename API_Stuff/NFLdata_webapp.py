import pickle
from flask import Flask, request, jsonify 

app = Flask(__name__)

def load_model():
    # unpickling
    infile = open("tree.p", "rb")
    header, tree = pickle.load(infile)
    infile.close()
    return header, tree

def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == "Leaf":
        return tree[1]
    att_index = header.index(tree[1])
    for i in range(2, len(tree)):
        value_list = tree[i]
        if value_list[1] == instance[att_index]:
            return tdidt_predict(header, value_list[2], instance)

# routes
@app.route("/")
def index():
    return "<h1> NFL predictions app </h1>", 200

@app.route("/predict")
def predict():
    WinPercentage = request.args.get("WinPercentage")
    RushYards = request.args.get("RushYards")
    PassYards = request.args.get("PassYards")
    Scoring = request.args.get("Scoring")
    RushYardsAllowed = request.args.get("RushYardsAllowed")
    PassYardsAllowed = request.args.get("PassYardsAllowed")
    DefenseScoringAllowed = request.args.get("DefenseScoringAllowed")
    KickingPercentage = request.args.get("KickingPercentage")
    TurnoverMargin = request.args.get("TurnoverMargin")
    instance = [WinPercentage, RushYards, PassYards, Scoring, RushYardsAllowed, PassYardsAllowed, DefenseScoringAllowed, KickingPercentage, TurnoverMargin]
    header, tree = load_model()
    pred = tdidt_predict(header, tree, instance)
    if pred is not None:
        return jsonify({"prediction": pred}), 200
    return "Error, could not make a prediction", 400


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5001, debug=True)
    # TODO: set debug to false before turning in project