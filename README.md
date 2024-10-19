# Translational Image-Guided Oncology: Coding Challenge

## Project Overview

This project develops a machine learning model to classify organ images based on radiomic features extracted from `.nrrd` files. The model processes a 15x15 pixel area of an organ, isolated from an MRI sequence, along with a binary mask specifying the region of interest (ROI). The output is a predicted label representing the organ.

### Contents
- `train_script.py`: A Python script to train the model and save it along with selected features.
- `predict_script.py`: A Python script to load the trained model and predict organ labels using new input images and masks.
- `final_model.pkl`: The saved model after training.
- `selected_features.json`: A file containing the list of features selected during training.
- `requirements.txt`: A file listing the required Python packages for running the scripts.
- `feature_correlation_matrix.png`: A heatmap showing the correlation matrix of the selected features.
- `lasso_alpha.png`: A plot showing the Mean Squared Error (MSE) across different lambda values in the LassoCV feature selection process.

## Installation

### Requirements
To run this project, you need to have Python installed, along with the required libraries listed in `requirements.txt`. You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```
The required libraries include:

```bash
joblib
numpy
pandas
scikit-learn
pynrrd
PyRadiomics
matplotlib
seaborn
statsmodels
```

## How to Train the Model

1. Ensure that you have directories containing your `.nrrd` image files and mask files.
2. Run the `train_script.py` to train the model. This script will:
   - Load the images and masks.
   - Extract radiomic features from the images using PyRadiomics.
   - Perform feature selection using Lasso regression and Variance Inflation Factor (VIF).
   - Train a Voting Classifier using Random Forest, AdaBoost, SVM, and KNN classifiers.
   - Save the trained model as `final_model.pkl` and the selected features as `selected_features.json`.

### Usage

```bash
python train_script.py
```

This will generate the following output files:

- `final_model.pkl`: The trained model.
- `selected_features.json`: The features selected during training.

It will also generate the following visualization files:

- `feature_correlation_matrix.png`: Displays the correlation matrix of the selected features.
- `lasso_alpha.png`: Shows the MSE for different lambda values in LassoCV.

## How to Make Predictions

- Ensure that you have the trained model (`final_model.pkl`) and the selected features (`selected_features.json`).
- Place your new `.nrrd` image and mask files in the appropriate directories.
- Run the `predict_script.py` to make predictions on new data. By default, the script uses the directories `data/images` and `data/masks`. You can specify different directories using the command-line arguments.

### Usage

To use default paths:

```bash
python predict_script.py
```
To specify custom paths for images and masks:
```bash
python predict_script.py --images_dir "path_to_images" --masks_dir "path_to_masks"
```
This will output:

- Predicted labels for the new input images.
- The actual labels (if they are extractable from filenames).
- The prediction accuracy.
```bash
Predicted labels: ['bonel' 'bonel' 'kidney']
Actual labels: ['bonel' 'bonel' 'kidney']
Prediction accuracy: 1.0000
```

## Selected Features

During training, the following features were selected and saved in `selected_features.json`:

- original_firstorder_Skewness
- original_firstorder_TotalEnergy
- original_glcm_ClusterShade
- original_glcm_Idn
- original_glcm_MCC
- original_gldm_LowGrayLevelEmphasis
- original_glszm_SizeZoneNonUniformity
- original_glszm_ZoneVariance
- original_ngtdm_Complexity

## The Task

The goal of the coding challenge is for you to __develop a basic machine learning model__ that takes as inputs two `.nrrd` files (`.nrrd` is a common file format in medical image computation, see [some pointers](#some-pointers)) and generates as output a single label.

- The first _input_ NRRD-file is a 15x15 pixel area of an organ, isolated from an MRI sequence. 
- The second _input_ NRRD-file is a 15x15 binary mask, specifying the region of interest (ROI) in the first file.
- The _output_ should be the name of the organ to which the first file belonged.

Your final submission should contain all the code you used to train your model, your final model, any code you used to generate visualizations. Moreover, please include an executable script/ function that loads your final model and makes predictions and reports accuracy for any two directories containing, respectively, the first and second input `.nrrd` files, as described above. 

In total, we intend for you to spend __no more than about 4 hours__ on the coding challenge.

Some other considerations:

- Start by cloning this repository. 
- When you are done, push the code and the results of your work. 
- The coding challenge should be completed using Python 3. You can use Jupyter Notebooks or standard Python scripts.
- We encourage the use of common machine learning packages, e.g., [scikit-learn](https://scikit-learn.org/stable/), [keras](https://keras.io/) or [pytorch](https://pytorch.org/). 
- Please include a `requirements.txt` in your submission, so that we can reproduce your code.
- Should you want to make use of a GPU-based machine learning method, you are encouraged to do so. For instance, Google [Colab](https://pytorch.org/) offers free GPU resources.
 

## The Data

The data is part of this repository and can be found in `./data`. It is organized as follows:

```
data
+--- images
|        1_liver7.nrrd
|        ...
+--- masks
|        1_liver7.nrrd
|        ...
```

Associated images (first input file described above) and masks (second input file described above) are labeled identically. So, `./data/masks/1_liver7.nrrd` is the ROI for `./data/images/1_liver7.nrrd`. 

The first number in the file name,  so `1` in `1_liver7.nrrd`, is the site at which the MRI image patch was recorded. In total, there are 7 sites. You have access to the data of 6 sites. We will be evaluating the performance of your model on the data of site 4, which we held back.

The file name also contains the label of the target organ, i.e., the intended output of your model. In this case the target label is `liver`. In total there are 5 different labels, i.e classes: liver, kidney, spleen, muscle and bone.


## Some Pointers

- To load `.nrrd` files you can use [pynrrd](https://pypi.org/project/pynrrd/). It provides easy access to the data in the form of a numpy array. 
- One possible approach to the problem is to start by extracting radiomics features from the images using [pyradiomics](https://pyradiomics.readthedocs.io/en/latest/) and then to use some of those features to train a model.
- An easy way to visualize images and masks is with [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php).

## Acknowledgements
This challenge was prepared by `jacob.murray@uk-essen.de`

