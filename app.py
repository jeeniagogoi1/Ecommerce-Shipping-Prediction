import pickle
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load the preprocessor and trained model
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            # Collect input data from the form
            Warehouse_block = request.form["Warehouse_block"]
            Mode_of_Shipment = request.form["Mode_of_Shipment"]
            Customer_care_calls = request.form["Customer_care_calls"]
            Customer_rating = request.form["Customer_rating"]
            Cost_of_the_Product = request.form["Cost_of_the_Product"]
            Prior_purchases = request.form["Prior_purchases"]
            Product_importance = request.form["Product_importance"]
            Gender = request.form["Gender"]
            Discount_offered = request.form["Discount_offered"]
            Weight_in_gms = request.form["Weight_in_gms"]

            # Create a dataframe from the input data
            input_data = pd.DataFrame({
                'Warehouse_block': [Warehouse_block],
                'Mode_of_Shipment': [Mode_of_Shipment],
                'Customer_care_calls': [Customer_care_calls],
                'Customer_rating': [Customer_rating],
                'Cost_of_the_Product': [Cost_of_the_Product],
                'Prior_purchases': [Prior_purchases],
                'Product_importance': [Product_importance],
                'Gender': [Gender],
                'Discount_offered': [Discount_offered],
                'Weight_in_gms': [Weight_in_gms],
                'Total_Interaction': [float(Customer_care_calls) * float(Customer_rating)],
                'Cost_per_Weight': [float(Cost_of_the_Product) / float(Weight_in_gms)]
            })

            # Preprocess the input data
            preds = preprocessor.transform(input_data)

            # Predict and calculate probabilities
            xx = model.predict(preds)
            prob = model.predict_proba(preds)[0]
            reach = prob[1]

            prediction_text = f'There is a {reach * 100:.2f}% chance that your product will reach in time'
            return render_template("prediction.html", prediction=prediction_text)
        except Exception as e:
            return render_template("prediction.html", prediction=f"Error in prediction: {e}")
    return render_template("prediction.html")

@app.route('/services')
def services():
    return render_template("services.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True, port=4000)
