import numpy as np
from flask import Flask,request,render_template
import pickle
app=Flask(__name__)
model=pickle.load(open("model.pkl",'rb'))


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    float_features=[float(x) for x in request.form.values()]
    features=np.array(float_features)
    features=features.reshape(1,-1)
    prediction=model.predict(features)
    print(prediction)
    ans=""
    if prediction[0]==0:
        ans="The person is not get placed"
    else:
        ans="The person is get Placed"
    return render_template("index.html",prediction_text="{}".format(ans))

if __name__=="__main__":
    app.run(debug=True)