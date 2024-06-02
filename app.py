from flask import Flask, jsonify, request
import pandas as pd
import jsonschema
from jsonschema import validate
import os
from datetime import datetime
import json
import xgboost as xgb
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold

app = Flask(__name__)

with open("schema/prediction.json") as file:
    prediction_schema = json.load(file)

with open("schema/training.json") as file:
    training_schema = json.load(file)

with open(f"preprocessing/target_mean_encoding.json", "r") as file:
    encoding = json.load(file)

with open("preprocessing/columns_order.txt", "r") as file:
    order = [line.strip() for line in file]

with open("model/metrics.json", "r") as file:
    metrics = json.load(file)

model = joblib.load("model/xgboost_model.joblib")

model_blueprint = xgb.XGBClassifier(max_depth = 4,
                                    subsample = 0.3,
                                    n_estimators = 500,
                                    learning_rate = 0.05, #default lr = 0.3
                                    min_child_weight = 1.5,
                                    #reg_alpha = 0,
                                    #reg_lambda = 0,
                                    booster = "gbtree",
                                    objective = "binary:logistic",
                                    random_state = 42)

def schema_validation(data: list, schema: dict) -> bool:
    if not isinstance(data, list): return False
    for index, entry in enumerate(data):
        try:
            validate(instance=entry, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            print(f"Validation failed for item {index + 1}: {e}")
            return False
    return True

def target_mean_encoding(data: pd.Series, target: pd.Series) -> pd.Series:
    weight = 5
    data = data.astype(object)
    mean = target.mean()
    agg = target.groupby(data).agg(["count","mean"])
    counts = agg["count"]
    means = agg["mean"]
    smooth = (counts * means + weight * mean) / (counts + weight)
    column = data.map(smooth)
    return column
        
def dict_encoding(keys: pd.Series, values: pd.Series) -> dict:
    keys = keys.astype(str)
    enc_dict = dict(zip(keys.unique(), values.unique()))
    return enc_dict

def map_encoding(data: pd.Series, encoding: dict) -> pd.Series:
    column = data.copy()
    column = column.astype(str)
    column = column.map(encoding)
    column = fill_na(column, encoding)
    return column

def fill_na(data: pd.Series, encoding: dict) -> pd.Series:
    column = data.copy()
    if column.isnull().values.any():
        val = list(encoding.values())
        avg = sum(val) / len(val)
        column = column.fillna(avg)
    return column

def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    # Transforming date_of_birth to age
    data.date_of_birth = pd.to_datetime(data.date_of_birth)
    data.date_of_birth = data.date_of_birth.apply(lambda x: datetime.now().year - x.year - ((datetime.now().month, datetime.now().day) < (x.month, x.day)))
    data = data.rename(columns={"date_of_birth": "age"})

    # Transforming contact
    data.contact = data.contact.apply(lambda x: 'Cellular' if x.startswith("+30 69") else ('Telephone' if x.startswith("+30 2") else "Unknown"))

    # Binary features preprocessing
    mapping = {"Yes": 1, "No": 0}
    data.default = data.default.map(mapping)
    data.housing = data.housing.map(mapping)
    data.loan = data.loan.map(mapping)
    data.contact = data.contact.map({"Cellular": 1, "Telephone": 0})
    data.previous_outcome = data.previous_outcome.map({"Failure": 0, "Success": 1})

    # Numeric features binning
    data.age = data.age.apply(lambda x: "a5" if x > 60 else "a4" if x > 50 else "a3" if x > 40 else "a2" if x > 30 else "a1")
    data.balance = data.balance.apply(lambda x: "b5" if x > 10000 else "b4" if x > 5000 else "b3" if x > 1500 else "b2" if x > 500 else "b1")
    data.calls = data.calls.apply(lambda x: "c5" if x > 10 else "c4" if x > 5 else "c3" if x > 2 else "c2" if x > 0 else "c1")
    data.previous_calls = data.previous_calls.apply(lambda x: "pc5" if x>30 else "pc4" if x>15 else "pc3" if x>5 else "pc2" if x > 0 else "pc1")
    return data
    
@app.route('/prediction', methods=['POST'])
def prediction():
    try:
        # Validate dataset's schema
        if not schema_validation(request.json["data"], prediction_schema):
            return jsonify({"Error": "Validation error."}) 

        # Convert request data to pandas dataframe and preprocessing
        data = pd.DataFrame(request.json["data"])
        customer_id = data.id
        data = data.drop(columns=["id"])
        data = preprocessing(data)
        data["month"] = datetime.now().strftime('%B')

        # Apply feature's encodings to the dataset
        for column in data.columns:
            data[column] = map_encoding(data[column], encoding[column])
        
        # Make predictions
        data = data[order]
        y_pred = model.predict(data)
        y_pred = pd.Series(y_pred, name="prediction")
        result = pd.concat([customer_id, y_pred], axis=1)
        result = result.replace({0: "Negative", 1: "Positive"})
        response = result.to_dict(orient="records")

        return jsonify(response)
    except Exception as e:
        return jsonify({"Error": str(e)}), 400

@app.route('/training/new', methods=['POST'])
def training_new():
    try:
        # Check dataset's length
        if not len(request.json["data"]) > 999:
            return jsonify({"message": "Not enough data."})

        # Validate dataset's schema
        if not schema_validation(request.json["data"], prediction_schema):
            return jsonify({"message": "Validation error."})
        
        # Convert request data to pandas dataframe and preprocessing
        data = pd.DataFrame(request.json["data"])
        data["calls"] = data["calls"] - 1
        data = preprocessing(data)
        data["last_call"] = pd.to_datetime(data["last_call"])
        data["month"] = data["last_call"].dt.strftime("%B")
        
        # Split dataset
        target = data.outcome
        target = target.map({"Success": 1, "Failure": 0})
        data = data.drop(columns=["id","outcome","last_call"])

        # Create a dictionary with feature's encodings
        encoding_dict = {}
        for column in data.columns:
            column_encoding = target_mean_encoding(data[column], target)
            column_encoding_dictionary = dict_encoding(data[column], column_encoding)
            encoding_dict[column] = column_encoding_dictionary

        # Apply feature's encodings to the dataset
        for column in data.columns:
            data[column] = map_encoding(data[column], encoding_dict[column])
        
        # Comparing mean accuracy and standard deviation with default model's metrics
        data = data[order]
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model_blueprint, data, target, cv=kfold, scoring="accuracy")
        if metrics["mean"] > cv_scores.mean() or metrics["std"] < cv_scores.std():
            print("Model did not change.")
            return jsonify({"message": "Model did not change."})

        # Creating and saving new model, new model's metrics, and feature encodings dictionary
        model = model_blueprint.fit(data, target)
        joblib.dump(model, "model/xgboost_model.joblib")
        metrics["mean"] = cv_scores.mean()
        metrics["std"] = cv_scores.std()

        with open(f"model/metrics.json", "w") as file:
            json.dump(metrics, file, indent=4)
        
        with open(f"preprocessing/target_mean_encoding.json", "w") as file:
            json.dump(encoding_dict, file, indent=4)

        print("New model created.")
        return jsonify({"message": "New model created."})
    except Exception as e:
        return jsonify({"Error": str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
