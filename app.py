import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import flask


app = Flask(__name__)


#model = pickle.load(open("model.pkl","rb"))
#model = pickle.load(open("/Users/merterkansozen/Desktop/Python-VSC/Local_1_heroku/model/model.pkl","rb"))  #merdiven klasör yapısı kurulduğunda, model klasörünün altında model.pkl konulduğunda.
model = pickle.load(open("/Users/merterkansozen/Desktop/Python-VSC/Local_1_heroku/model.pkl","rb"))


@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    features = [str(x) for x in request.form.values()]

    final_features = [np.array(features)]
    prediction = model.predict_proba(final_features)

    output='{0:.{1}f}'.format(prediction[0][1], 2)


    return render_template('index.html',pred='Your probability of diabetes is % {}'.format(str(float(output)*100)))

if __name__ == "__main__":
 #   app.run(debug=True)                            # hiçbirşey yazmasan otomatik 127.0.0.1:50002e atar
 #   app.run(host='127.0.0.1',port=3000,debug=True)  # istersen böyle manuel ayarlayıp portu 3000 olarak ayarlayabilirsin
 #   app.run(host='192.168.2.101', port=3000)  # debug=True demezsen debug etmeden direk run eder, hataları göremezsin
    app.run(host='127.0.0.1',port=3000)