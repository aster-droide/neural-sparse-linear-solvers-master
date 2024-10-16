import numpy as np
import scipy.sparse
import scipy.linalg
import os

# model predictions on reduced test set
predictions_folder = 'C:\\Users\\astrid\\Downloads\\neural-sparse-linear-solvers-master\\predictions\\test'
batch_files = [f for f in os.listdir(predictions_folder) if f.endswith('.npz')]

# init vars to accumulate RMSE, max difference, and residuals
total_rmse = 0
total_max_diff = 0
total_scipy_residual = 0
num_batches = len(batch_files)

# it over each batch file & compute the metrics
for batch_file in batch_files:
    batch_file_path = os.path.join(predictions_folder, batch_file)

    batch_data = np.load(batch_file_path)

    # extract A_indices, A_values, model-generated x, and b
    A_indices = batch_data['A_indices']
    A_values = batch_data['A_values']
    x_model = batch_data['x_model']  # model-generated x (y_direction)
    b = batch_data['b']  # target vector b

    # reconstruct sparse matrix A using scipy.sparse
    A_shape = (b.size, b.size)
    A = scipy.sparse.coo_matrix((A_values, (A_indices[0], A_indices[1])), shape=A_shape)

    # convert a to dense for conversion
    A_dense = A.todense()

    # solve for x using Scipy
    x_scipy = scipy.linalg.solve(A_dense, b)

    # compute the difference between model-generated x and Scipy-computed x
    x_diff = x_model - x_scipy

    # calculate RMSE for Scipy-computed x
    rmse = np.sqrt(np.mean((x_diff) ** 2))

    # compute residual for Scipy-computed x
    b_scipy_predicted = A_dense @ x_scipy  # A * x_scipy
    scipy_residual = np.linalg.norm(b_scipy_predicted - b)

    # accumulate the metrics
    total_rmse += rmse
    total_scipy_residual += scipy_residual

    # individual results for each batch
    print(f"Results for {batch_file}:")
    print(f"  RMSE between model-generated x and Scipy-computed x: {rmse}")
    print(f"  Scipy residual: {scipy_residual}")
    print("\n")

# compute average RMSE, max absolute difference, and residuals
avg_rmse = total_rmse / num_batches
avg_scipy_residual = total_scipy_residual / num_batches

# display avgs
print(f"Average RMSE over all batches: {avg_rmse}")
print(f"Average Scipy residual over all batches: {avg_scipy_residual}")
