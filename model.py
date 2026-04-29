import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE


def train_model():
    data = pd.read_csv("healthcare-dataset-stroke-data.csv")

    data = data.drop(columns=["id"], errors="ignore")

    data["bmi"] = data["bmi"].fillna(data["bmi"].mean())

    y = data["stroke"]
    X = data.drop("stroke", axis=1)

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y )

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight={0: 1, 1: 6}
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds)
    }

    return model, metrics, X.columns, data