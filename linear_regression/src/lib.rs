mod linear_regression;

struct RegressionData {
    //using f64 although f32 is probably good for most cases
    feature_matrix: DMatrix<f64>, //The feature matrix to be used as training data
    target_vector: DVector<f64>,  //The target vector
}

struct ModelWeights {
    model_weights: DVector<f64>, //vector to store model weights
}

struct LinearRegressor {
    data: RegressionData,
    model: ModelWeights,
}


