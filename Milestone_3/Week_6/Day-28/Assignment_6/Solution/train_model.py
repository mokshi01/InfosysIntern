# train_model.py
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = "iris_model.joblib"
RANDOM_STATE = 42

def main():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")

    # Save model with extra metadata
    joblib.dump(
        {
            "model": clf,
            "feature_names": iris.feature_names,
            "target_names": iris.target_names,
        },
        MODEL_PATH,
    )
    print(f"Model saved as {MODEL_PATH}")

if __name__ == "__main__":
    main()
