import matplotlib.pyplot as plt
# Visualizing the dataset
def plot_feature_vs_target(x_train, y_train, x_cv, y_cv, x_test, y_test):

  # Assuming X_train, y_train, X_cv, y_cv, X_test, y_test are your training,
  # cross-validation, and test sets respectively

  # Assuming feature_names is a list of the names of your features
  feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Add more features according to your dataset']

  # Plot each feature against the target variable
  num_features = x_train.shape[1]
  fig, axs = plt.subplots(num_features, figsize=(8, 6 * num_features))

  for i in range(num_features):
      axs[i].scatter(x_train[:, i], y_train, label='Training Set')
      axs[i].scatter(x_cv[:, i], y_cv, label='Cross-Validation Set')
      axs[i].scatter(x_test[:, i], y_test, label='Test Set')
      axs[i].set_title(f'{feature_names[i]} vs Target')
      axs[i].set_xlabel(feature_names[i])
      axs[i].set_ylabel('Target')
      axs[i].legend()

  plt.tight_layout()
  plt.show()
plot_feature_vs_target(x_train, y_train, x_cv, y_cv, x_test, y_test)

def plot_feature_vs_target(x_train, y_train, y_preds_train, x_cv=None, y_cv=None, y_preds_cv=None, x_test=None, y_test=None, y_preds_test=None):
    # Assuming feature_names is a list of the names of your features
    feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']

    # Plot each feature against the target variable
    num_features = x_train.shape[1]
    fig, axs = plt.subplots(num_features, figsize=(8, 6 * num_features))

    for i in range(num_features):
        axs[i].scatter(x_train[:, i], y_train, label='Training Set (Actual)')
        axs[i].scatter(x_train[:, i], y_preds_train, label='Training Set (Predicted)', marker='o')
        
        if x_cv is not None and y_cv is not None:
            axs[i].scatter(x_cv[:, i], y_cv, label='Cross-Validation Set (Actual)')
            if y_preds_cv is not None:
                axs[i].scatter(x_cv[:, i], y_preds_cv, label='Cross-Validation Set (Predicted)', marker='x')
        
        if x_test is not None and y_test is not None:
            axs[i].scatter(x_test[:, i], y_test, label='Test Set (Actual)')
            if y_preds_test is not None:
                axs[i].scatter(x_test[:, i], y_preds_test, label='Test Set (Predicted)', marker='^')

        axs[i].set_title(f'{feature_names[i]} vs Target')
        axs[i].set_xlabel(feature_names[i])
        axs[i].set_ylabel('Target')
        axs[i].legend()

    plt.tight_layout()
    plt.show()

plot_feature_vs_target(x_train, y_train, y_preds_train, x_cv, y_cv, y_preds_cv, x_test, y_test, y_preds_test)