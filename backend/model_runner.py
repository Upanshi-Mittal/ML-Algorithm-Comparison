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
    X = df.drop(columns=[target])
    y = df[target]

    preprocessor = build_pipeline(X)

    # Train-test split (raw data for sklearn pipelines)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),

        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Naive Bayes": GaussianNB(),

        "Gradient Boosting": GradientBoostingClassifier(),
        "Extra Trees": ExtraTreesClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }

    results = []

    # 1. Run sklearn models
    for name, model in models.items():
        try:
            pipe = Pipeline(steps=[
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            acc = accuracy_score(y_test, preds)

            results.append({
                "model": name,
                "accuracy": round(acc, 4)
            })

        except Exception as e:
            results.append({
                "model": name,
                "error": str(e)
            })

    # 2. Run PyTorch ANN (separately)
    try:
        X_processed = preprocessor.fit_transform(X)

        X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )

        ann_acc = train_ann(X_train_p, y_train_p, X_test_p, y_test_p)

        results.append({
            "model": "PyTorch ANN",
            "accuracy": ann_acc
        })

    except Exception as e:
        results.append({
            "model": "PyTorch ANN",
            "error": str(e)
        })

    # Sort results
    results = sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True)

    results = results[:5]
    return results