==============================================================================
CUSTOM LOGISTIC REGRESSION & K-FOLD VALIDATION (NUMPY IMPLEMENTATION)
==============================================================================

OVERVIEW
--------
This project provides a "from-scratch" implementation of Logistic Regression 
built entirely using NumPy. It is designed for binary classification tasks and 
includes advanced features often found in deep learning frameworks, such as 
adaptive optimizers (Adam, RMSProp) and L1 regularization.

Additionally, a utility function for K-Fold cross-validation is included to 
assist in dataset splitting.

------------------------------------------------------------------------------
TABLE OF CONTENTS
------------------------------------------------------------------------------
1. Requirements
2. Features
3. Class Documentation: LogisticRegression
4. Function Documentation: K_fold
5. Usage Example

------------------------------------------------------------------------------
1. REQUIREMENTS
------------------------------------------------------------------------------
To run this code, you need Python installed along with the following libraries:

- numpy: For matrix operations and math functions.
- tqdm:  For progress bars during training loops.

Installation:
    pip install numpy tqdm

------------------------------------------------------------------------------
2. FEATURES
------------------------------------------------------------------------------
- **Pure NumPy:** No dependency on Scikit-Learn or PyTorch.
- **Custom Optimizers:** Supports three modes of gradient descent:
    1. "None": Standard Stochastic/Mini-batch Gradient Descent.
    2. "rms":  RMSProp (Root Mean Square Propagation).
    3. "adam": Adam (Adaptive Moment Estimation).
- **Regularization:** Built-in L1 regularization (Lasso) to prevent overfitting.
- **Mini-Batch Support:** Capable of training on full datasets or mini-batches.
- **Progress Tracking:** Real-time console progress bar showing loss/accuracy.

------------------------------------------------------------------------------
3. CLASS DOCUMENTATION: LogisticRegression
------------------------------------------------------------------------------

Initialization:
    model = LogisticRegression(learning_rate=0.05, maxIter=1000, ...)

Parameters:
    - learning_rate (float): Step size for gradient descent (default: 0.05).
    - maxIter (int): Maximum number of training epochs (default: 1000).
    - error_ratio (float): Threshold to stop training early if error is low.
    - L1 (float): L1 regularization term (default: 0).
    - batch_size (int): Size of data batches. Set to 1 for SGD (default: 1).
    - beta_1 (float): Decay rate for first moment estimates [Adam] (default: 0.9).
    - beta_2 (float): Decay rate for second moment estimates [Adam/RMS] (default: 0.9).
    - epsilon (float): Small value to prevent division by zero (default: 0.5).

Methods:

    fit(X, Y, optimizer="None")
    ---------------------------
    Trains the model on the provided data.
    - X: Input features (numpy array).
    - Y: Target labels (numpy array).
    - optimizer: String. Options are "None", "adam", or "rms".

    predict(X_test)
    ---------------
    Performs binary classification on new data.
    - X_test: Input features.
    - Returns: Array of binary predictions (0 or 1).

    evaluate(X_test, Y_test)
    ------------------------
    Calculates the accuracy of the model.
    - Returns: Float (Accuracy percentage, e.g., 0.95).

    get_weights() / set_weights(weights)
    ------------------------------------
    Getters and setters for the model coefficients.

------------------------------------------------------------------------------
4. FUNCTION DOCUMENTATION: K_fold
------------------------------------------------------------------------------

    K_fold(data, K=5)

Description:
    Splits the dataset into K folds for cross-validation.

Parameters:
    - data: The complete dataset (rows = samples).
    - K: Number of splits (default: 5).

Returns:
    - A 3D numpy array containing the folded data.

IMPORTANT NOTE ON SHAPE:
    The current implementation of K_fold assumes a specific input width.
    - It initializes arrays with a hardcoded dimension of 785 columns.
    - It reshapes the output to (10, -1, 785).
    *Ensure your input data matches these dimensions (e.g., MNIST with bias) 
    or modify the hardcoded values in the function before use.*

------------------------------------------------------------------------------
5. USAGE EXAMPLE
------------------------------------------------------------------------------

    import numpy as np
    from your_module import LogisticRegression, K_fold

    # 1. Generate Dummy Data (100 samples, 2 features)
    X = np.random.rand(100, 2)
    Y = np.random.randint(0, 2, 100)

    # 2. Initialize Model
    # Using Adam optimizer settings and a batch size of 10
    model = LogisticRegression(
        learning_rate=0.01, 
        maxIter=50, 
        batch_size=10
    )

    # 3. Train
    print("Training Model...")
    model.fit(X, Y, optimizer="adam")

    # 4. Evaluate
    accuracy = model.evaluate(X, Y)
    print(f"Final Accuracy: {accuracy * 100:.2f}%")

    # 5. Predict new value
    sample = np.array([[0.5, 0.5]])
    prediction = model.predict(sample)
    print(f"Prediction for {sample}: {prediction}")

==============================================================================
