import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ann_model import train_ann

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib

def detect_problem(y):
    unique_vals = len(set(y))
    if unique_vals <= 15:
        return "classification"
    else:
        return "regression"

def build_pipeline(X):
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor


def run_models(df, target):
    from sklearn.preprocessing import LabelEncoder

    X = df.drop(columns=[target])
    y = df[target]

    problem_type = detect_problem(y)

    if problem_type == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)

    preprocessor = build_pipeline(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem_type == "classification":
        models = {
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "Gradient Boosting": GradientBoostingClassifier(),
        }
        metric_name = "accuracy"
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
        }
        metric_name = "r2_score"

    results = []
    trained_models = {}

    for name, model in models.items():
        try:
            pipe = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            if problem_type == "classification":
                score = accuracy_score(y_test, preds)
            else:
                score = r2_score(y_test, preds)

            results.append({
                "model": name,
                metric_name: round(score, 4)
            })

            trained_models[name] = pipe

        except Exception as e:
            results.append({
                "model": name,
                "error": str(e)
            })

    key = metric_name
    results = sorted(results, key=lambda x: x.get(key, 0), reverse=True)

    best_model_name = results[0]["model"]
    best_model = trained_models.get(best_model_name)

    if best_model:
        joblib.dump(best_model, "best_model.pkl")

    explanation = f"{best_model_name} performed best because it handled the dataset patterns effectively."

    print("FINAL RESULTS:", results)
    print("TRAINED MODELS:", trained_models.keys())
    return {
        "type": problem_type,
        "best_model": best_model_name,
        "explanation": explanation,
        "results": results[:5]
    }