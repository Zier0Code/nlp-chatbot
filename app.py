from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from Chatbot import pred_class, get_response, words, classes, data_filtered

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    msg = request.get_json().get("message")
    if not msg:
        return jsonify({"answer": "No message provided"}), 400

    ints = pred_class(msg, words, classes)
    res = {"response": get_response(ints,data_filtered),
           "message": msg,
           "tag": ints[0]}
    return jsonify(res)

print(classes)
if __name__ == '__main__':
    app.run(debug=True)