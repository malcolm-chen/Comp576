import numpy as np
import scipy.linalg

# MATLAB: ndims(a)
a = np.array([[1, 2, 3], [4, 5, 6]])
ndims_a = np.ndim(a)  # or a.ndim
# Number of dimensions of array a

# MATLAB: numel(a)
numel_a = np.size(a)  # or a.size
# Number of elements in array a

# MATLAB: size(a)
size_a = np.shape(a)  # or a.shape
# "Size" of array a

# MATLAB: size(a, n)
size_a_n = a.shape[1]  # For the n-th dimension, note the 0-based indexing in Python

# MATLAB: [ 1 2 3; 4 5 6 ]
b = np.array([[1., 2., 3.], [4., 5., 6.]])
# Define a 2x3 2D array

# MATLAB: [ a b; c d ]
c = np.array([[7., 8., 9.]])
d = np.block([[b, c], [b, c]])
# Construct a matrix from blocks a, b, c, d

# MATLAB: a(end)
end_elem = a[-1]
# Access the last element of a

# MATLAB: a(2, 5)
elem = a[1, 4]  # Second row, fifth column (Python uses 0-based indexing)

# MATLAB: a(2,:)
second_row = a[1]  # or a[1, :]
# Entire second row of a

# MATLAB: a(1:5,:)
first_5_rows = a[0:5]  # or a[:5]
# First five rows of 2D array a

# MATLAB: a(end-4:end,:)
last_5_rows = a[-5:]
# Last five rows of 2D array a

# MATLAB: a(1:3,5:9)
sub_matrix = a[0:3, 4:9]
# First through third rows and fifth through ninth columns of a 2D array a

# MATLAB: a([2,4,5],[1,3])
sub_matrix_rows_cols = a[np.ix_([1, 3, 4], [0, 2])]
# Access rows 2,4,5 and columns 1 and 3

# MATLAB: a(3:2:21,:)
every_other_row = a[2:21:2, :]
# Every other row from the third to the twenty-first

# MATLAB: a(1:2:end,:)
every_other_row_from_first = a[::2, :]
# Every other row starting from the first

# MATLAB: a(end:-1:1,:) or flipud(a)
reversed_a = a[::-1, :]
# Reverse rows of array a

# MATLAB: a([1:end 1],:)
appended_first_row = np.r_[:len(a), 0]
# Append the first row to the end of array a

# MATLAB: a.'
transpose_a = a.T  # or a.transpose()
# Transpose of a

# MATLAB: a'
conjugate_transpose = a.conj().T  # or a.conj().transpose()
# Conjugate transpose of a

# MATLAB: a * b (matrix multiplication)
matrix_multiply = a @ b
# Matrix multiplication of a and b

# MATLAB: a .* b (element-wise multiplication)
element_wise_multiply = a * b
# Element-wise multiplication of a and b

# MATLAB: a./b (element-wise division)
element_wise_divide = a / b
# Element-wise division of a by b

# MATLAB: a.^3 (element-wise exponentiation)
element_wise_exponentiation = a ** 3
# Element-wise exponentiation of a

# MATLAB: find(a > 0.5)
find_gt_05 = np.nonzero(a > 0.5)
# Find the indices where a > 0.5

# MATLAB: a(a<0.5)=0
a[a < 0.5] = 0
# Set elements of a < 0.5 to zero

# MATLAB: y=x
y = a.copy()
# NumPy assigns by reference, so use `.copy()` to get a copy of array a

# MATLAB: y=x(:)
flattened_a = a.flatten()
# Turn array into a vector

# MATLAB: 1:10
arange_1_to_10 = np.arange(1, 11)
# Create an increasing vector from 1 to 10

# MATLAB: [1:10]'
column_vector = np.arange(1., 11.)[:, np.newaxis]
# Create a column vector

# MATLAB: zeros(3,4)
zeros_array = np.zeros((3, 4))
# Create a 3x4 array of zeros

# MATLAB: ones(3,4)
ones_array = np.ones((3, 4))
# Create a 3x4 array of ones

# MATLAB: eye(3)
identity_matrix = np.eye(3)
# 3x3 identity matrix

# MATLAB: diag(a)
diag_elements = np.diag(a)
# Get diagonal elements of array a

# MATLAB: diag(v,0)
diag_matrix = np.diag(b, 0)
# Create a diagonal matrix from vector v

# MATLAB: max(max(a))
max_a = a.max()
# Maximum element of a

# MATLAB: max(a)
max_columns = a.max(0)
# Maximum element of each column in array a

# MATLAB: norm(v)
norm_v = np.linalg.norm(a)
# L2 norm of vector a

