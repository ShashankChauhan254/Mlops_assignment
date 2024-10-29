import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

data_path = "BostonHousing.csv"
data = pd.read_csv(data_path)

X = data.drop(columns=['medv']) 
y = data['medv']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear_Regression': LinearRegression(),
    'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

mlflow.set_experiment("Housing_Price_Models")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)

        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name} - MSE: {mse}")
