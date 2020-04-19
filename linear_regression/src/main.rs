extern crate nalgebra;
use nalgebra::base::{DMatrix, DVector};

fn main() {}

struct RegressionData {
    //using f64 although f32 is probably good for most cases
    feature_matrix: DMatrix<f64>, //The feature matrix to be used as training data
    target_vector: DVector<f64>,  //The target vector
}

struct ModelWeights {
    model_weights: DVector<f64>, //vector to store model weights
}

fn predict(feature_matrix: &DMatrix<f64>, model_weights: &DVector<f64>) -> DVector<f64> {
    feature_matrix * model_weights //make a prediction
}

fn sum_squared_errors(
    feature_matrix: &DMatrix<f64>,
    model_weights: &DVector<f64>,
    target_vector: &DVector<f64>,
) -> f64 {
    let length_of_target = target_vector.len() as f64;
    //make a prediction using the model weights and the feature matrix
    let prediction_vector = predict(feature_matrix, target_vector);
    let difference = prediction_vector - target_vector;
    //compute the sum of the squared errors divided by twice the number of training points
    difference.norm_squared() / (2.0 * length_of_target)
}

fn sum_squared_errors_gradient(
    feature_matrix: &DMatrix<f64>,
    model_weights: &DVector<f64>,
    target_vector: &DVector<f64>,
) -> DVector<f64> {
    //Compute the gradient of the sum of squared errors divided by twice the number of training points
    let length_of_target = target_vector.len() as f64;
    let inverse_twice_length_of_target: f64 = 1.0 / (2.0 * length_of_target);
    let diagonal_square_matrix: DMatrix<f64> = DMatrix::from_diagonal_element(
        target_vector.len(),
        target_vector.len(),
        inverse_twice_length_of_target,
    );
    let error: DVector<f64> = feature_matrix * model_weights - target_vector;
    diagonal_square_matrix * (error.transpose() * feature_matrix).transpose()
}

fn gradient_descent(
    feature_matrix: &DMatrix<f64>,
    target_vector: &DVector<f64>,
    learning_rate: &f32,
    max_iterations: &u32,
) -> DVector<f64> {
    let length_of_target = target_vector.len() as f64;
    let inverse_length_of_target: f64 = 1.0 / length_of_target;
    let diagonal_square_matrix: DMatrix<f64> = DMatrix::from_diagonal_element(
        target_vector.len(),
        target_vector.len(),
        inverse_length_of_target,
    );
    let initial_model = diagonal_square_matrix * feature_matrix.row_sum_tr();
    initial_model
}
