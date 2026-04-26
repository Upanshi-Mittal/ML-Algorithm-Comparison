import pandas as pd
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, r2_score,
    classification_report, confusion_matrix
)

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor
)
from sklearn.svm import SVC

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ------------------ Detect Problem Type ------------------
def detect_problem(y):
    return "classification" if len(set(y)) <= 15 else "regression"


# ------------------ Build Preprocessing Pipeline ------------------
def build_pipeline(X):
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor


# ------------------ Main Function ------------------
def run_models(df, target):

    # -------- Dataset Analysis --------
    analysis = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing": df.isnull().sum().to_dict()
    }

    X = df.drop(columns=[target])
    y = df[target]

    problem_type = detect_problem(y)

    # Encode target for classification
    if problem_type == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)

    preprocessor = build_pipeline(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- Model Selection --------
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

    # -------- Train Models --------
    for name, model in models.items():
        try:
            start = time.time()

            pipe = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            end = time.time()

            if problem_type == "classification":
                score = accuracy_score(y_test, preds)
            else:
                score = r2_score(y_test, preds)

            results.append({
                "model": name,
                metric_name: round(score, 4),
                "time": round(end - start, 3)
            })

            trained_models[name] = pipe

        except Exception as e:
            results.append({
                "model": name,
                metric_name: 0,
                "error": str(e)
            })

    # -------- Sort Results --------
    key = metric_name
    results = sorted(results, key=lambda x: x.get(key, 0), reverse=True)

    valid_results = [r for r in results if key in r]

    if not valid_results:
        return {
            "type": problem_type,
            "results": [],
            "analysis": analysis,
            "explanation": "All models failed"
        }

    best_model_name = valid_results[0]["model"]
    best_model = trained_models.get(best_model_name)

    # -------- Save Best Model --------
    if best_model:
        joblib.dump(best_model, "best_model.pkl")

    feature_importance = None

    try:
        model = best_model.named_steps["model"]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

            # get transformed feature names
            feature_names = best_model.named_steps["preprocessing"].get_feature_names_out()

            feature_importance = [
                {
                    "feature": str(f),
                    "importance": float(i)
                }
                for f, i in zip(feature_names, importances)
            ]

    except Exception as e:
        print("Feature importance error:", e)

    # -------- Classification Metrics --------
    report = None
    conf_matrix = None

    if problem_type == "classification" and best_model:
        preds = best_model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, preds).tolist()

        scores = [r.get(key) for r in valid_results]

        if len(set(scores)) == 1:
            explanation = "All models performed equally well on this dataset."
        else:
            if problem_type == "classification":
                explanation = f"{best_model_name} performed best because it captures non-linear patterns and feature interactions effectively."
            else:
                explanation = f"{best_model_name} performed best by fitting the numerical relationships in the dataset efficiently."

    # -------- Final Output --------
    return {
        "type": problem_type,
        "best_model": best_model_name,
        "explanation": explanation,
        "results": valid_results[:5],
        "analysis": analysis,
        "feature_importance": feature_importance,
        "classification_report": report,
        "confusion_matrix": conf_matrix
    }