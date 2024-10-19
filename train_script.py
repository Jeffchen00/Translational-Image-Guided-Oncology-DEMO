import os
import warnings
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import json
from radiomics import featureextractor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')


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

    for filename in sorted(os.listdir(images_dir)):
        if filename.endswith('.nrrd'):
            feature_vector = extractor.execute(os.path.join(images_dir, filename), os.path.join(masks_dir, filename))
            features.append(list(feature_vector.values()))

            if feature_names is None:
                feature_names = list(feature_vector.keys())

    features_df = pd.DataFrame(features, columns=feature_names)
    features_df = features_df.filter(regex='^(?!diagnostics)', axis=1)

    return features_df


def feature_selection_with_pearson_corrilation(X, threshold=0.85):
    """
    Perform feature selection by removing highly correlated features using Pearson correlation.

    Args:
        X (pd.DataFrame): DataFrame of features.
        threshold (float): Threshold for Pearson correlation to identify correlated features (default 0.85).

    Returns:
        pd.Index: Index of selected feature names that remain after correlation-based filtering.
    """
    corr_matrix = X.corr().abs()

    while True:
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        if not to_drop:
            break
        X = X.drop(columns=[to_drop[0]])

        corr_matrix = X.corr().abs()

    return X.columns


def feature_selection_with_vif(X, threshold=5):
    """
    Perform feature selection using Variance Inflation Factor (VIF) to remove multicollinear features.

    Args:
        X (pd.DataFrame): DataFrame of features.
        threshold (float): Threshold value for VIF to remove features (default 5).

    Returns:
        pd.Index: Index of selected feature names that remain after VIF-based filtering.
    """
    X_scaled = StandardScaler().fit_transform(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

    while True:
        max_vif = vif_data["VIF"].max()
        if max_vif <= threshold:
            break

        feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
        X = X.drop(columns=[feature_to_remove])
        X_scaled = StandardScaler().fit_transform(X)
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

    return vif_data["Feature"]


def draw_correlation_matrix(X):
    """
    Draw and save a heatmap of the correlation matrix for the provided features.

    Args:
        X (pd.DataFrame): DataFrame of features.
    """
    f, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', annot_kws={'size': 10, 'weight': 'bold', }, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, va="top", ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    plt.savefig('feature_correlation_matrix.png')


def feature_selection_with_lasso(X, labels):
    """
    Perform feature selection using Lasso regression (LassoCV).

    Args:
        X (pd.DataFrame): DataFrame of features.
        labels (np.array): Array of labels.

    Returns:
        pd.Index: Index of selected feature names that remain after Lasso-based filtering.
    """
    def draw_lasso_alpha(lasso_cv):
        """
        Draw and save a plot of the LassoCV alpha path and MSE values.

        Args:
            lasso_cv (LassoCV): Trained LassoCV model.
        """
        MSEs = lasso_cv.mse_path_
        mse = list()
        std = list()
        for m in MSEs:
            mse.append(np.mean(m))
            std.append(np.std(m))

        plt.figure(figsize=(8, 6))
        plt.errorbar(lasso_cv.alphas_, mse, std, fmt='o:', ecolor='lightblue', elinewidth=3, ms=5, mfc='wheat',
                     mec='salmon', capsize=3)
        plt.axvline(lasso_cv.alpha_, color='red', ls='--')
        plt.title('Errorbar')
        plt.xlabel('Lambda')
        plt.ylabel('MSE')
        plt.savefig('lasso_alpha.png')

    label_encoder = LabelEncoder()
    y_numeric = label_encoder.fit_transform(labels)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply LassoCV for feature selection
    alpha_range = np.logspace(-3, -2, 50, base=5)
    lasso_cv = LassoCV(alphas=alpha_range, cv=10, max_iter=5000)
    lasso_cv.fit(X_scaled, y_numeric)

    # Get the selected features
    lasso_coefficients = lasso_cv.coef_
    selected_features = X.columns[lasso_coefficients != 0]

    draw_lasso_alpha(lasso_cv)

    return selected_features


def train(X, y):
    """
    Train multiple classifiers (RandomForest, AdaBoost, SVM, KNN) and use a soft voting classifier.

    Args:
        X (pd.DataFrame): DataFrame of selected features.
        y (np.array): Array of labels.

    Returns:
        VotingClassifier: Trained voting classifier.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

    param_grids = {'Random Forest': {'model': RandomForestClassifier(class_weight='balanced'),
                                     'params': {'n_estimators': [200], 'max_depth': [None, 10, 20],
                                                'min_samples_split': [2, 5, 10]}},
                   'AdaBoost': {'model': AdaBoostClassifier(algorithm='SAMME'),
                                'params': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 1.0, 2.0]}},
                   'SVM': {'model': SVC(probability=True, class_weight='balanced'),
                           'params': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}},
                   'KNN': {'model': KNeighborsClassifier(),
                           'params': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}}}

    best_estimators = []

    for name, clf_info in param_grids.items():
        model = clf_info['model']
        param_grid = clf_info['params']

        grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', n_jobs=-1)

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_estimators.append((name, best_model))

        y_test_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print(f"{name}: Test Accuracy: {test_accuracy:.4f}")

    voting_clf = VotingClassifier(estimators=best_estimators, voting='soft')
    voting_clf.fit(X_train, y_train)

    y_test_pred = voting_clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Voting Classifier Test Accuracy: {test_accuracy:.4f}")

    return voting_clf


def train_and_save_model(images_dir, masks_dir, model_filename='final_model.pkl', features_filename='selected_features.json'):
    """
    Train the model using images and masks from the given directories and save the final model and selected features.

    Args:
        images_dir (str): Directory containing NRRD image files.
        masks_dir (str): Directory containing NRRD mask files.
        model_filename (str): Filename for saving the trained model (default 'final_model.pkl').
        features_filename (str): Filename for saving the selected feature names (default 'selected_features.json').

    Returns:
        VotingClassifier: The trained voting classifier.
    """
    # Step 1: Load the data
    images, masks, labels = load_nrrd_files(images_dir, masks_dir)

    # Step 2: Extract features
    X = extract_features(images_dir, masks_dir)

    # Step 3: Feature selection using Lasso and VIF
    selected_features = feature_selection_with_lasso(X, labels)
    selected_features = feature_selection_with_vif(X[selected_features], threshold=5)

    # Step 4: Draw and save correlation matrix
    draw_correlation_matrix(X[selected_features])

    # Step 5: Save selected feature names to a JSON file
    with open(features_filename, 'w') as f:
        json.dump(list(selected_features), f)
    print(f"Selected features saved to {features_filename}")

    # Step 6: Train the model
    clf = train(X[selected_features], labels)

    # Step 7: Save the trained model to disk
    joblib.dump(clf, model_filename)
    print(f"Model saved to {model_filename}")

    return clf


if __name__ == '__main__':
    # Specify your image and mask directories here
    images_dir = 'data/images'
    masks_dir = 'data/masks'

    # Train the model and save it
    clf = train_and_save_model(images_dir, masks_dir)
