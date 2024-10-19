import os
import numpy as np
import joblib
import nrrd
import pandas as pd
import json
import argparse
import logging
from radiomics import featureextractor
from sklearn.metrics import accuracy_score


def load_nrrd_files(images_dir, masks_dir):
    """
    Load NRRD image and mask files from specified directories and extract corresponding labels.

    Args:
        images_dir (str): Directory containing NRRD image files.
        masks_dir (str): Directory containing NRRD mask files.

    Returns:
        tuple: A tuple containing three elements:
            - np.array: A NumPy array of images.
            - np.array: A NumPy array of masks.
            - np.array: A NumPy array of labels extracted from the file names.
    """
    images = []
    masks = []
    labels = []

    for filename in sorted(os.listdir(images_dir)):
        if filename.endswith('.nrrd'):
            image_path = os.path.join(images_dir, filename)
            mask_path = os.path.join(masks_dir, filename)

            # Load the image and the mask
            image_data, _ = nrrd.read(image_path)
            mask_data, _ = nrrd.read(mask_path)

            # Extract label from the filename (e.g., 1_liver7.nrrd -> liver)
            label = filename.split('_')[1][:-6]

            images.append(image_data)
            masks.append(mask_data)
            labels.append(label)

    return np.array(images), np.array(masks), np.array(labels)


def extract_features(images_dir, masks_dir):
    """
    Extract radiomic features from NRRD image and mask files using PyRadiomics.

    Args:
        images_dir (str): Directory containing NRRD image files.
        masks_dir (str): Directory containing NRRD mask files.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted features, with feature names as columns.
    """
    extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True)
    features = []
    feature_names = None


    # set level for all classes
    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)

    for filename in sorted(os.listdir(images_dir)):
        if filename.endswith('.nrrd'):
            feature_vector = extractor.execute(os.path.join(images_dir, filename), os.path.join(masks_dir, filename))
            features.append(list(feature_vector.values()))

            if feature_names is None:
                feature_names = list(feature_vector.keys())

    features_df = pd.DataFrame(features, columns=feature_names)
    features_df = features_df.filter(regex='^(?!diagnostics)', axis=1)

    return features_df


def predict_new_data(images_dir, masks_dir, model_filename='final_model.pkl',
                     features_filename='selected_features.json'):
    """
    Load a saved model and make predictions on new image and mask files.

    Args:
        images_dir (str): Directory containing NRRD image files.
        masks_dir (str): Directory containing NRRD mask files.
        model_filename (str): Filename for the saved model (default 'final_model.pkl').
        features_filename (str): Filename for the saved selected feature names (default 'selected_features.json').

    Returns:
        None
    """
    # Load the trained model
    clf = joblib.load(model_filename)
    print(f"Model loaded from {model_filename}")

    # Load the selected features from file
    with open(features_filename, 'r') as f:
        selected_features = json.load(f)
    print(f"Selected features loaded from {features_filename}")

    # Load and preprocess the new data
    images, masks, labels = load_nrrd_files(images_dir, masks_dir)
    X = extract_features(images_dir, masks_dir)

    # Filter the features to only include the selected ones
    X_selected = X[selected_features]

    # Make predictions
    y_pred = clf.predict(X_selected)

    # Print the predicted and actual labels
    print(f"Predicted labels: {y_pred}")
    print(f"Actual labels: {labels}")

    # Calculate and report accuracy
    accuracy = accuracy_score(labels, y_pred)
    print(f"Prediction accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Predict organ labels from NRRD images and masks.')

    # Add arguments for the image and mask directories, with defaults
    parser.add_argument('--images_dir', type=str, default='data/images',
                        help='Directory containing NRRD image files. Default is "data/images".')
    parser.add_argument('--masks_dir', type=str, default='data/masks',
                        help='Directory containing NRRD mask files. Default is "data/masks".')

    # Parse the arguments
    args = parser.parse_args()

    # Call the predict_new_data function with the provided or default directories
    predict_new_data(images_dir=args.images_dir, masks_dir=args.masks_dir)
