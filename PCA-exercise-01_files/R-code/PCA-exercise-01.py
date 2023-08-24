# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:40.393621Z","iopub.execute_input":"2023-08-23T20:17:40.394074Z","iopub.status.idle":"2023-08-23T20:17:40.405428Z","shell.execute_reply.started":"2023-08-23T20:17:40.394037Z","shell.execute_reply":"2023-08-23T20:17:40.404253Z"}}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 150) # displays all columns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# %% [markdown]
# # Dimensionality Reduction - PCA

# %% [markdown]
# # 01 - Data Preparation

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.495279Z","iopub.execute_input":"2023-08-23T20:17:41.495633Z","iopub.status.idle":"2023-08-23T20:17:41.507444Z","shell.execute_reply.started":"2023-08-23T20:17:41.495596Z","shell.execute_reply":"2023-08-23T20:17:41.506214Z"}}
# Load the training data from the specified address
train_original = pd.read_csv('/kaggle/input/dfc-state/DFC_STATE.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.509949Z","iopub.execute_input":"2023-08-23T20:17:41.510409Z","iopub.status.idle":"2023-08-23T20:17:41.528485Z","shell.execute_reply.started":"2023-08-23T20:17:41.510375Z","shell.execute_reply":"2023-08-23T20:17:41.526772Z"}}
# Create a working copy of the original DataFrame
train_data = train_original.copy()

# Print basic dataset info
train_data.info()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.532186Z","iopub.execute_input":"2023-08-23T20:17:41.532614Z","iopub.status.idle":"2023-08-23T20:17:41.545688Z","shell.execute_reply.started":"2023-08-23T20:17:41.532580Z","shell.execute_reply":"2023-08-23T20:17:41.544514Z"}}
train_data.columns

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.546858Z","iopub.execute_input":"2023-08-23T20:17:41.547194Z","iopub.status.idle":"2023-08-23T20:17:41.584512Z","shell.execute_reply.started":"2023-08-23T20:17:41.547163Z","shell.execute_reply":"2023-08-23T20:17:41.583040Z"}}
train_data.head(3)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.586491Z","iopub.execute_input":"2023-08-23T20:17:41.586906Z","iopub.status.idle":"2023-08-23T20:17:41.602548Z","shell.execute_reply.started":"2023-08-23T20:17:41.586870Z","shell.execute_reply":"2023-08-23T20:17:41.600511Z"}}
# Drop null records
train_data = train_data.dropna()

# Drop unecessary features
#train_data = train_data.drop(['ID'], axis=1)

# Check for null values
train_data.isnull().sum()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.605008Z","iopub.execute_input":"2023-08-23T20:17:41.605929Z","iopub.status.idle":"2023-08-23T20:17:41.636103Z","shell.execute_reply.started":"2023-08-23T20:17:41.605885Z","shell.execute_reply":"2023-08-23T20:17:41.634734Z"}}
# Check duplicated rows, or records

train_data.loc[train_data.duplicated()]

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.638103Z","iopub.execute_input":"2023-08-23T20:17:41.638475Z","iopub.status.idle":"2023-08-23T20:17:41.748844Z","shell.execute_reply.started":"2023-08-23T20:17:41.638444Z","shell.execute_reply":"2023-08-23T20:17:41.747919Z"}}
# Descriptive statistics for numerical features
train_data.describe()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.750109Z","iopub.execute_input":"2023-08-23T20:17:41.750753Z","iopub.status.idle":"2023-08-23T20:17:41.764997Z","shell.execute_reply.started":"2023-08-23T20:17:41.750693Z","shell.execute_reply":"2023-08-23T20:17:41.763278Z"}}
# Descriptive statistics for categorical features
train_data.describe(include = "object")

# %% [markdown]
# # 02 - EDA

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.766799Z","iopub.execute_input":"2023-08-23T20:17:41.767108Z","iopub.status.idle":"2023-08-23T20:17:41.783214Z","shell.execute_reply.started":"2023-08-23T20:17:41.767080Z","shell.execute_reply":"2023-08-23T20:17:41.780900Z"}}
# import ydata_profiling

# # Create EDA report
# eda_report = ydata_profiling.ProfileReport(train_original)
# #eda_report

# # Export the report to a file
# eda_report.to_file("dialysis_eda_report.html")

# %% [markdown]
# # 03 - Feature Engineering
# ## Create the X, Y matrices

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.785266Z","iopub.execute_input":"2023-08-23T20:17:41.785764Z","iopub.status.idle":"2023-08-23T20:17:41.802482Z","shell.execute_reply.started":"2023-08-23T20:17:41.785724Z","shell.execute_reply":"2023-08-23T20:17:41.800805Z"}}
# Define features X-matrix
X = train_data.drop(['Survival- As Expected (STATE)', 'State'], axis=1)

# Define target y-matrix
y = train_data['Survival- As Expected (STATE)']

# Records comparison for the X-matrix, Y-matrix
train_data.shape, X.shape, y.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.803712Z","iopub.execute_input":"2023-08-23T20:17:41.804037Z","iopub.status.idle":"2023-08-23T20:17:41.838315Z","shell.execute_reply.started":"2023-08-23T20:17:41.804007Z","shell.execute_reply":"2023-08-23T20:17:41.836738Z"}}
X.head()

# %% [markdown]
# ## Feature Scaling

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.842973Z","iopub.execute_input":"2023-08-23T20:17:41.843327Z","iopub.status.idle":"2023-08-23T20:17:41.856721Z","shell.execute_reply.started":"2023-08-23T20:17:41.843294Z","shell.execute_reply":"2023-08-23T20:17:41.854193Z"}}
# Standardization
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# %% [markdown]
# ## Multiple Linear Regression - Full Model

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.858307Z","iopub.execute_input":"2023-08-23T20:17:41.860041Z","iopub.status.idle":"2023-08-23T20:17:41.887310Z","shell.execute_reply.started":"2023-08-23T20:17:41.859979Z","shell.execute_reply":"2023-08-23T20:17:41.885929Z"}}
random_state = 95
X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size = 0.3,
                     shuffle = True,
                     random_state=random_state)
#---train the model using Multiple Linear Regression---
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
#---evaluate the model---
linear_reg.score(X_test,y_test)

# %% [markdown]
# ## PCA to X-matrix

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.889459Z","iopub.execute_input":"2023-08-23T20:17:41.889997Z","iopub.status.idle":"2023-08-23T20:17:41.911236Z","shell.execute_reply.started":"2023-08-23T20:17:41.889958Z","shell.execute_reply":"2023-08-23T20:17:41.909485Z"}}
from sklearn.decomposition import PCA
components = None
pca = PCA(n_components = components)
# perform PCA on the scaled data
pca.fit(X_scaled)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.913269Z","iopub.execute_input":"2023-08-23T20:17:41.914369Z","iopub.status.idle":"2023-08-23T20:17:41.927734Z","shell.execute_reply.started":"2023-08-23T20:17:41.914282Z","shell.execute_reply":"2023-08-23T20:17:41.926084Z"}}
# print the explained variances
print("Variances (Percentage):")
print(pca.explained_variance_ratio_ * 100)
print()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.929869Z","iopub.execute_input":"2023-08-23T20:17:41.930827Z","iopub.status.idle":"2023-08-23T20:17:41.938915Z","shell.execute_reply.started":"2023-08-23T20:17:41.930776Z","shell.execute_reply":"2023-08-23T20:17:41.937969Z"}}
print("Cumulative Variances (Percentage):")
print(pca.explained_variance_ratio_.cumsum() * 100)
print()

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:41.940315Z","iopub.execute_input":"2023-08-23T20:17:41.940934Z","iopub.status.idle":"2023-08-23T20:17:42.193631Z","shell.execute_reply.started":"2023-08-23T20:17:41.940893Z","shell.execute_reply":"2023-08-23T20:17:42.192081Z"}}
import matplotlib.pyplot as plt

# scree plot
components = len(pca.explained_variance_ratio_) \
    if components is None else components
plt.plot(range(1,components+1), 
         np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")

# %% [markdown]
# <font size='5'>Let’s now apply PCA to find the desired number of components based on the desired explained variance.<font>

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:42.195113Z","iopub.execute_input":"2023-08-23T20:17:42.195554Z","iopub.status.idle":"2023-08-23T20:17:42.423960Z","shell.execute_reply.started":"2023-08-23T20:17:42.195510Z","shell.execute_reply":"2023-08-23T20:17:42.422845Z"}}
pca = PCA(n_components = 0.86)
pca.fit(X_scaled)
print("Cumulative Variances (Percentage):")
print(np.cumsum(pca.explained_variance_ratio_ * 100))
components = len(pca.explained_variance_ratio_)
print(f'Number of components: {components}')
# Make the scree plot
plt.plot(range(1, components + 1), np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel("Number of components")
plt.ylabel("Explained variance (%)")

# %% [markdown]
# <font size='5'>You can also find out the importance of each feature that contributes to each of the components using the components_ attribute of the pca object<font>

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:42.425371Z","iopub.execute_input":"2023-08-23T20:17:42.425705Z","iopub.status.idle":"2023-08-23T20:17:42.434618Z","shell.execute_reply.started":"2023-08-23T20:17:42.425675Z","shell.execute_reply":"2023-08-23T20:17:42.432937Z"}}
pca_components = abs(pca.components_)
print(pca_components)

# %% [markdown]
# <font size='5'>The importance of each feature is reflected by the magnitude of the corresponding values in the output — the higher magnitude, the higher the importance.<font>

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:42.436419Z","iopub.execute_input":"2023-08-23T20:17:42.436853Z","iopub.status.idle":"2023-08-23T20:17:42.455067Z","shell.execute_reply.started":"2023-08-23T20:17:42.436814Z","shell.execute_reply":"2023-08-23T20:17:42.453078Z"}}
print('Top 4 most important features in each component')
print('===============================================')
for row in range(pca_components.shape[0]):
    # get the indices of the top 4 values in each row
    temp = np.argpartition(-(pca_components[row]), 4)
    
    # sort the indices in descending order
    indices = temp[np.argsort((-pca_components[row])[temp])][:4]
    
    # print the top 4 feature names
    print(f'Component {row}: {train_data.columns[indices].to_list()}')

# %% [markdown]
# ## Transforming all the Columns to the (n) Principal Components

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:42.457106Z","iopub.execute_input":"2023-08-23T20:17:42.457600Z","iopub.status.idle":"2023-08-23T20:17:42.472434Z","shell.execute_reply.started":"2023-08-23T20:17:42.457558Z","shell.execute_reply":"2023-08-23T20:17:42.470877Z"}}
# transform the standardized data of the columns in the dataset to the (n) principal components
X_pca = pca.transform(X_scaled)
print(X_pca.shape)
print(X_pca)

# %% [markdown]
# # 04 - Modeling
# ## Create Pipeline

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:42.474197Z","iopub.execute_input":"2023-08-23T20:17:42.474818Z","iopub.status.idle":"2023-08-23T20:17:42.481161Z","shell.execute_reply.started":"2023-08-23T20:17:42.474783Z","shell.execute_reply":"2023-08-23T20:17:42.479858Z"}}
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


_sc = StandardScaler()
_pca = PCA(n_components = components)
_model = LinearRegression()
linear_reg_model = Pipeline([
    ('std_scaler', _sc),
    ('pca', _pca),
    ('regressor', _model)
])

# %% [markdown]
# ## Data split into Training, and Test sets

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:42.482176Z","iopub.execute_input":"2023-08-23T20:17:42.482473Z","iopub.status.idle":"2023-08-23T20:17:42.499460Z","shell.execute_reply.started":"2023-08-23T20:17:42.482445Z","shell.execute_reply":"2023-08-23T20:17:42.497562Z"}}
# perform a split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=random_state)


# %% [markdown]
# ## Train, and Predict using Pipeline

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:42.501172Z","iopub.execute_input":"2023-08-23T20:17:42.501509Z","iopub.status.idle":"2023-08-23T20:17:42.523510Z","shell.execute_reply.started":"2023-08-23T20:17:42.501478Z","shell.execute_reply":"2023-08-23T20:17:42.522438Z"}}
# train the model using the PCA components
linear_reg_model.fit(X_train,y_train)

# Use the pipeline to make predictions
y_pred = linear_reg_model.predict(X_test)

# %% [markdown]
# ## Model Evaluation

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:42.525188Z","iopub.execute_input":"2023-08-23T20:17:42.525962Z","iopub.status.idle":"2023-08-23T20:17:42.534313Z","shell.execute_reply.started":"2023-08-23T20:17:42.525928Z","shell.execute_reply":"2023-08-23T20:17:42.532721Z"}}
import math

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared:", r2)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:42.536121Z","iopub.execute_input":"2023-08-23T20:17:42.536794Z","iopub.status.idle":"2023-08-23T20:17:42.550183Z","shell.execute_reply.started":"2023-08-23T20:17:42.536755Z","shell.execute_reply":"2023-08-23T20:17:42.548757Z"}}
# Calculate the adjusted R-squared
n = len(y)  # Number of observations
p = len(X.columns)  # Number of predictors (independent variables/features)
adjusted_r_squared = 1 - ((1 - r2) * (n - 1)/ (n - p - 1))

print(n, p)
print("Adjusted R-squared:", adjusted_r_squared)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:42.551472Z","iopub.execute_input":"2023-08-23T20:17:42.551804Z","iopub.status.idle":"2023-08-23T20:17:42.569930Z","shell.execute_reply.started":"2023-08-23T20:17:42.551776Z","shell.execute_reply":"2023-08-23T20:17:42.568521Z"}}
# Extract coefficients and intercept from the model
coefficients = _model.coef_
intercept = _model.intercept_

# Print coefficients and intercept
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# %% [markdown]
# # 05 - Kernel PCA

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:42.571294Z","iopub.execute_input":"2023-08-23T20:17:42.572507Z","iopub.status.idle":"2023-08-23T20:17:42.595828Z","shell.execute_reply.started":"2023-08-23T20:17:42.572464Z","shell.execute_reply":"2023-08-23T20:17:42.594632Z"}}
from sklearn.decomposition import KernelPCA

# Split the data into training and testing sets
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Create a Kernel PCA transformer
kpca = KernelPCA(n_components=8, kernel='linear')

# Create a linear regression model
multiple_reg_model = LinearRegression()

# Create a pipeline that combines Kernel PCA and Linear Regression
kernel_pipeline = Pipeline([
    ('kpca', kpca),
    ('regressor', multiple_reg_model)
])

# Fit the pipeline to the training data
kernel_pipeline.fit(X_train_k, y_train_k)

# Make predictions on the test data
y_pred_k = kernel_pipeline.predict(X_test_k)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:42.600040Z","iopub.execute_input":"2023-08-23T20:17:42.600613Z","iopub.status.idle":"2023-08-23T20:17:42.613409Z","shell.execute_reply.started":"2023-08-23T20:17:42.600566Z","shell.execute_reply":"2023-08-23T20:17:42.612099Z"}}
# Evaluate the model
mse = mean_squared_error(y_test_k, y_pred_k)
r2 = r2_score(y_test_k, y_pred_k)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:42.617715Z","iopub.execute_input":"2023-08-23T20:17:42.618515Z","iopub.status.idle":"2023-08-23T20:17:42.629172Z","shell.execute_reply.started":"2023-08-23T20:17:42.618474Z","shell.execute_reply":"2023-08-23T20:17:42.628397Z"}}
# Calculate the adjusted R-squared
n = len(y)  # Number of observations
p = len(X.columns)  # Number of predictors (independent variables/features)
adjusted_r_squared = 1 - ((1 - r2) * (n - 1)/ (n - p - 1))

print(n, p)
print("Adjusted R-squared:", adjusted_r_squared)

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:42.633008Z","iopub.execute_input":"2023-08-23T20:17:42.635956Z","iopub.status.idle":"2023-08-23T20:17:42.645138Z","shell.execute_reply.started":"2023-08-23T20:17:42.635912Z","shell.execute_reply":"2023-08-23T20:17:42.644323Z"}}
# Extract coefficients and intercept from the model
coefficients = multiple_reg_model.coef_
intercept = multiple_reg_model.intercept_

# Print coefficients and intercept
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# %% [markdown]
# # 06 - Graphs

# %% [code] {"execution":{"iopub.status.busy":"2023-08-23T20:17:42.649365Z","iopub.execute_input":"2023-08-23T20:17:42.650150Z","iopub.status.idle":"2023-08-23T20:17:42.970606Z","shell.execute_reply.started":"2023-08-23T20:17:42.650111Z","shell.execute_reply":"2023-08-23T20:17:42.968424Z"}}
import matplotlib.pyplot as plt

# Perform Kernel PCA on the training data
X_train_kpca = kpca.transform(X_train_k)

# Plot the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_train_kpca[:, 0], X_train_kpca[:, 1], c=y_train_k, cmap=plt.cm.Paired)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Kernel PCA - First Two Principal Components')
plt.colorbar(label='Target Value')
plt.show()
plt.close()
