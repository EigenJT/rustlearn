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
    learning_rate: f64,
    tolerance: f64,
    max_iterations: u32,
) -> DVector<f64> {
    let length_of_target = target_vector.len() as f64;
    let inverse_length_of_target = 1.0 / length_of_target;
    let model_size = feature_matrix.shape().1;
    //create diagonal matrix to divide sums by
    let diagonal_inverse_number_of_points_matrix: DMatrix<f64> =
        DMatrix::from_diagonal_element(model_size, model_size, inverse_length_of_target);
    //initialize model to the average of all the features. Need to take inverse of model
    let mut model: DVector<f64> =
        diagonal_inverse_number_of_points_matrix * feature_matrix.row_sum_tr();
    //create a vector of all ones
    let all_ones = DVector::from_element(model_size, 1.0f64);
    //divide it by the model to create the new model
    model = all_ones.component_div(&model);
    //create diagonal learning rate matrix
    let diagonal_learning_rate_matrix: DMatrix<f64> =
        DMatrix::from_diagonal_element(model_size, model_size, learning_rate);
    //initialize loss
    let mut error: f64 = sum_squared_errors(&feature_matrix, &model, &target_vector);
    //initialize number of iterations
    let mut n_iterations: u32 = 0;
    //set final model to loop until either tolerance or max iterations is met
    let final_model: DVector<f64> = loop {
        //increment iterations
        n_iterations += 1;
        //set model to new model using gradient descent
        model = &model
            - &diagonal_learning_rate_matrix
                * sum_squared_errors_gradient(&feature_matrix, &model, &target_vector);
        //compute error
        error = sum_squared_errors(&feature_matrix, &model, &target_vector);
        //if error < tolerance, break the loop and return the model
        if error < tolerance {
            break model;
        };
        // break the loop and return the model if the maximum number of iterations is reached
        if max_iterations == n_iterations {
            println!("Maximum ({}) iterations reached!", max_iterations);
            break model;
        };
    };
    //return the model
    final_model
}
