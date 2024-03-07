from flask import Flask , render_template, jsonify, request
import pickle

scaler_model = pickle.load(open("models/scaler1.pkl", "rb"))
log_model = pickle.load(open("models/log_reg.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/prediction", methods=['GET','POST'])
def prediction():
    if request.method =='POST':
        Pregnancies = int(request.form.get("Pregnancies"))
        Glucose = int(request.form.get("Glucose"))
        BloodPressure = int(request.form.get("BloodPressure"))
        SkinThickness = int(request.form.get("SkinThickness"))
        Insulin = int(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
        Age = int(request.form.get("Age"))
    else:
        return render_template("home.html")

    new_scaled_data = scaler_model.transform([[Pregnancies, Glucose, BloodPressure ,SkinThickness,Insulin, BMI,DiabetesPedigreeFunction	,Age]])
    result = log_model.predict(new_scaled_data)


    return render_template("home.html", result=result)

if __name__=="__main__":
    app.run(host="0.0.0.0")
