import joblib
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from src.aki.config import SETTINGS


def main():
    """
    Converts the trained XGBoost pickle into ONNX format.
    Ensures input type matches the 10-feature vector expected by the model.
    """
    pkl_path = SETTINGS.MODEL_PATH.with_suffix(".pkl")
    onnx_path = SETTINGS.MODEL_PATH.with_suffix(".onnx")

    print(f"Loading {pkl_path}...")
    clf = joblib.load(pkl_path)

    # strip feature names to prevent ONNX version mismatches
    clf.get_booster().feature_names = None

    # input: [batch_size, 10 features]
    initial_type = [('float_input', FloatTensorType([None, 10]))]

    print("Converting...")
    onnx_model = onnxmltools.convert_xgboost(clf, initial_types=initial_type)

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Model saved to {onnx_path}.")


if __name__ == "__main__":
    main()