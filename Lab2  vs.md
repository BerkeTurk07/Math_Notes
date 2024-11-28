
# Lab 2: Matrix Operations

## 1. Basic Operations on Matrices

### Given Matrices:
A = `[[1, 2], [3, 4]]`  
B = `[[5, 6], [7, 8]]`  
C = `[[-1, 2], [3, 0]]`  
D = `[[-1, 2, 3], [4, 0, 6]]`  
E = `[[1, 2], [4, 5], [7, 8]]`  

### Results:

#### 1.1 Addition and Subtraction:
- A + B = Matrix([[6, 8], [10, 12]])
- B - A = Matrix([[4, 4], [4, 4]])
- A + C = Matrix([[0, 4], [6, 4]])
- **D - E is not defined due to incompatible dimensions**

#### 1.2 Scalar Multiplication:
- (1/3) * A = Matrix([[1/3, 2/3], [1, 4/3]])
- 2 * B = Matrix([[10, 12], [14, 16]])
- -3 * C = Matrix([[3, -6], [-9, 0]])
- 4 * D = Matrix([[-4, 8, 12], [16, 0, 24]])

#### 1.3 Matrix Products:
- A * B = Matrix([[19, 22], [43, 50]])
- B * A = Matrix([[23, 34], [31, 46]])
- A * D = Matrix([[7, 2, 15], [13, 6, 33]])
- D * E = Matrix([[28, 32], [46, 56]])
  

## 2. Determinants 2x2 and 3x3

### Determinants for 2x2 Matrices:
- **det(A)** = det([[2, 3], [1, 4]]) = (2 * 4) - (3 * 1) = 8 - 3 = **5**
- **det(B)** = det([[5, 6], [7, 8]]) = (5 * 8) - (6 * 7) = 40 - 42 = **-2**
- **det(C)** = det([[-1, 2], [3, 0]]) = (-1 * 0) - (2 * 3) = 0 - 6 = **-6**

### Determinants for 3x3 Matrices:
- **det(D)** = det([[-1, 2, 3], [4, 0, 6], [1, 2, 4]])  
  Using cofactor expansion on the first row:  
  det(D) = -1 * det([[0, 6], [2, 4]]) - 2 * det([[4, 6], [1, 4]]) + 3 * det([[4, 0], [1, 2]])  
  det(D) = -1 * ((0 * 4) - (6 * 2)) - 2 * ((4 * 4) - (6 * 1)) + 3 * ((4 * 2) - (0 * 1))  
  det(D) = -1 * (-12) - 2 * (16 - 6) + 3 * (8)  
  det(D) = 12 - 2 * 10 + 24  
  det(D) = **-18**

- **det(E)** = det([[1, 2, 4], [5, 3, 2], [7, 8, 5]])  
  Using cofactor expansion on the first row:  
  det(E) = 1 * det([[3, 2], [8, 5]]) - 2 * det([[5, 2], [7, 5]]) + 4 * det([[5, 3], [7, 8]])  
  det(E) = 1 * ((3 * 5) - (2 * 8)) - 2 * ((5 * 5) - (2 * 7)) + 4 * ((5 * 8) - (3 * 7))  
  det(E) = 1 * (15 - 16) - 2 * (25 - 14) + 4 * (40 - 21)  
  det(E) = -1 - 2 * 11 + 4 * 19  
  det(E) = **-5**

- **det(F)** = det([[2, -3, 1], [1, 4, -2], [1, 5, 3]])  
  Using cofactor expansion on the first row:  
  det(F) = 2 * det([[4, -2], [5, 3]]) - (-3) * det([[1, -2], [1, 3]]) + 1 * det([[1, 4], [1, 5]])  
  det(F) = 2 * ((4 * 3) - (-2 * 5)) + 3 * ((1 * 3) - (-2 * 1)) + 1 * ((1 * 5) - (4 * 1))  
  det(F) = 2 * (12 + 10) + 3 * (3 + 2) + 1 * (5 - 4)  
  det(F) = 2 * 22 + 3 * 5 + 1 * 1  
  det(F) = **0**

### Results:
- **det(A)** = 5
- **det(B)** = -2
- **det(C)** = -6
- **det(D)** = -18
- **det(E)** = -5
- **det(F)** = 0


## 3. Determinants using Laplace's Expansion

### Given Matrices:
- A = `[[2, 3, 1], [4, 0, 1], [3, 2, 1]]`
- B = `[[2, 3, 1], [4, 1, 0], [3, 2, 0]]`
- C = `[[2, 3, 1], [1, 0, 0], [3, 2, 0]]`
- D = `[[2, 3, 1, 4], [5, 3, 2, 0], [1, 4, 3, 2], [1, 4, 2, 0]]`

### Results:

#### 3.1 Determinant of A using Laplace Expansion:
Using cofactor expansion on the first row:
- det(A) = 2 * det([[0, 1], [2, 1]]) - 3 * det([[4, 1], [3, 1]]) + 1 * det([[4, 0], [3, 2]])  
det(A) = 2 * ((0 * 1) - (1 * 2)) - 3 * ((4 * 1) - (1 * 3)) + 1 * ((4 * 2) - (0 * 3))  
det(A) = 2 * (-2) - 3 * (4 - 3) + 1 * (8)  
det(A) = -4 - 3 + 8  
det(A) = **1**

#### 3.2 Determinant of B using Laplace Expansion:
Using cofactor expansion on the first row:
- det(B) = 2 * det([[1, 0], [2, 0]]) - 3 * det([[4, 0], [3, 0]]) + 1 * det([[4, 1], [3, 2]])  
det(B) = 2 * ((1 * 0) - (0 * 2)) - 3 * ((4 * 0) - (0 * 3)) + 1 * ((4 * 2) - (1 * 3))  
det(B) = 2 * (0) - 3 * (0) + 1 * (8 - 3)  
det(B) = 0 + 0 + 5  
det(B) = **5**

#### 3.3 Determinant of C using Laplace Expansion:
Using cofactor expansion on the first row:
- det(C) = 2 * det([[0, 0], [2, 0]]) - 3 * det([[1, 0], [3, 0]]) + 1 * det([[1, 0], [3, 2]])  
det(C) = 2 * ((0 * 0) - (0 * 2)) - 3 * ((1 * 0) - (0 * 3)) + 1 * ((1 * 2) - (0 * 3))  
det(C) = 2 * (0) - 3 * (0) + 1 * (2)  
det(C) = 0 + 0 + 2  
det(C) = **2**

#### 3.4 Determinant of D using Laplace Expansion:
Using cofactor expansion on the first row:
- det(D) = 2 * det([[3, 2, 0], [4, 3, 2], [4, 2, 0]]) - 3 * det([[5, 2, 0], [1, 3, 2], [1, 2, 0]]) + 1 * det([[5, 3, 2], [1, 4, 2], [1, 4, 0]])  
det(D) = 2 * ((3 * 3 * 0) + (2 * 2 * 4) + (0 * 4 * 2) - (2 * 3 * 4) - (2 * 3 * 1) - (0 * 3 * 1))  
det(D) = 2 * (0 + 16 + 0 - 24 - 6 + 0)  
det(D) = 2 * (-14)  
det(D) = **-28**


## 4. Inverse of Matrices

### Given Matrices:
- A = `[[2, 1], [3, 4]]`
- B = `[[1, 2], [3, 4]]`
- C = `[[2, 3, 1], [4, 0, 1], [3, 2, 1]]`

### Results:

#### 4.1 Inverse of A:
The inverse of a 2x2 matrix can be found using the formula:  
\[ A^{-1} = \frac{1}{det(A)} \cdot adj(A) \]  
Where \( adj(A) \) is the adjugate of matrix A.

- det(A) = (2 * 4) - (1 * 3) = 8 - 3 = **5**
- adj(A) = `[[4, -1], [-3, 2]]`

So, the inverse of A is:
\[ A^{-1} = \frac{1}{5} \cdot [[4, -1], [-3, 2]] = [[4/5, -1/5], [-3/5, 2/5]] \]

#### 4.2 Inverse of B:
- det(B) = (1 * 4) - (2 * 3) = 4 - 6 = **-2**
- adj(B) = `[[4, -2], [-3, 1]]`

So, the inverse of B is:
\[ B^{-1} = \frac{1}{-2} \cdot [[4, -2], [-3, 1]] = [[-2, 1], [3/2, -1/2]] \]

#### 4.3 Inverse of C:
For a 3x3 matrix, the inverse can be calculated using the formula:
\[ C^{-1} = \frac{1}{det(C)} \cdot adj(C) \]  
Where \( adj(C) \) is the adjugate of matrix C.

- det(C) = 2 * det([[0, 1], [2, 1]]) - 3 * det([[4, 1], [3, 1]]) + 1 * det([[4, 0], [3, 2]])  
det(C) = 2 * ((0 * 1) - (1 * 2)) - 3 * ((4 * 1) - (1 * 3)) + 1 * ((4 * 2) - (0 * 3))  
det(C) = 2 * (-2) - 3 * (4 - 3) + 1 * (8)  
det(C) = -4 - 3 + 8 = **1**

Since det(C) = 1, C is invertible.

The adjugate of C is:
\[ adj(C) = [[-2, -1, 2], [-2, 1, 4], [2, -3, 1]] \]

So, the inverse of C is:
\[ C^{-1} = \frac{1}{1} \cdot [[-2, -1, 2], [-2, 1, 4], [2, -3, 1]] = [[-2, -1, 2], [-2, 1, 4], [2, -3, 1]] \]


## 5. Eigenvalues and Eigenvectors

### Given Matrices:
- A = `[[4, -2], [1, 1]]`
- B = `[[1, 2], [2, 1]]`
- C = `[[2, 0], [0, 3]]`

### Results:

#### 5.1 Eigenvalues and Eigenvectors of A:
To find the eigenvalues of A, we solve the characteristic equation:
\[ det(A - \lambda I) = 0 \]

For matrix A, we solve:
\[ det([[4, -2], [1, 1]] - \lambda [[1, 0], [0, 1]]) = 0 \]

This simplifies to:
\[ det([[4 - \lambda, -2], [1, 1 - \lambda]]) = 0 \]

The characteristic equation becomes:
\[
(4 - \lambda)(1 - \lambda) - (-2)(1) = 0
\]
\[
\lambda^2 - 5\lambda + 6 = 0
\]

Solving for λ:
\[
\lambda = \frac{5 \pm \sqrt{25 - 24}}{2} = \frac{5 \pm 1}{2}
\]
So, the eigenvalues of A are:
\[
\lambda_1 = 3, \quad \lambda_2 = 2
\]

To find the eigenvectors, we substitute each eigenvalue into \( (A - \lambda I)v = 0 \).

For \( \lambda_1 = 3 \):
\[
(A - 3I)v = 0 \Rightarrow \begin{bmatrix} 1 & -2 \\ 1 & -2 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = 0
\]
Solving this system gives the eigenvector:
\[
v_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}
\]

For \( \lambda_2 = 2 \):
\[
(A - 2I)v = 0 \Rightarrow \begin{bmatrix} 2 & -2 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = 0
\]
Solving this system gives the eigenvector:
\[
v_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
\]

#### 5.2 Eigenvalues and Eigenvectors of B:
For matrix B, we solve the characteristic equation:
\[ det(B - \lambda I) = 0 \]

We get:
\[
det([[1 - \lambda, 2], [2, 1 - \lambda]]) = 0
\]

This simplifies to:
\[
(1 - \lambda)(1 - \lambda) - (2)(2) = 0
\]
\[
\lambda^2 - 2\lambda - 3 = 0
\]

Solving for λ:
\[
\lambda = \frac{2 \pm \sqrt{4 + 12}}{2} = \frac{2 \pm \sqrt{16}}{2} = \frac{2 \pm 4}{2}
\]
So, the eigenvalues of B are:
\[
\lambda_1 = 3, \quad \lambda_2 = -1
\]

For \( \lambda_1 = 3 \):
\[
(B - 3I)v = 0 \Rightarrow \begin{bmatrix} -2 & 2 \\ 2 & -2 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = 0
\]
Solving this system gives the eigenvector:
\[
v_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
\]

For \( \lambda_2 = -1 \):
\[
(B + I)v = 0 \Rightarrow \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = 0
\]
Solving this system gives the eigenvector:
\[
v_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}
\]

#### 5.3 Eigenvalues and Eigenvectors of C:
For matrix C, we solve the characteristic equation:
\[ det(C - \lambda I) = 0 \]

We get:
\[
det([[2 - \lambda, 0], [0, 3 - \lambda]]) = 0
\]

This simplifies to:
\[
(2 - \lambda)(3 - \lambda) = 0
\]

So, the eigenvalues of C are:
\[
\lambda_1 = 2, \quad \lambda_2 = 3
\]

For \( \lambda_1 = 2 \):
\[
(C - 2I)v = 0 \Rightarrow \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = 0
\]
Solving this system gives the eigenvector:
\[
v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
\]

For \( \lambda_2 = 3 \):
\[
(C - 3I)v = 0 \Rightarrow \begin{bmatrix} -1 & 0 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = 0
\]
Solving this system gives the eigenvector:
\[
v_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}


## 6. Matrix Diagonalization

### Given Matrices:
- A = `[[4, -2], [1, 1]]`
- B = `[[1, 2], [2, 1]]`
- C = `[[2, 0], [0, 3]]`

### Results:

#### 6.1 Diagonalization of A:
Matrix A is diagonalizable if there exists an invertible matrix P such that:
\[ A = P \cdot D \cdot P^{-1} \]
Where D is a diagonal matrix of eigenvalues, and P is a matrix whose columns are the corresponding eigenvectors.

From the previous section, we know the eigenvalues of A are \( \lambda_1 = 3 \) and \( \lambda_2 = 2 \), and the corresponding eigenvectors are:
- \( v_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix} \)
- \( v_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix} \)

Thus, matrix P is formed by placing the eigenvectors as columns:
\[ P = \begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix} \]

The diagonal matrix D is:
\[ D = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix} \]

To find \( P^{-1} \), we compute the inverse of P:
\[ P^{-1} = \frac{1}{\text{det}(P)} \cdot \text{adj}(P) \]

- det(P) = (2 * 1) - (1 * 1) = 1
- adj(P) = `[[1, -1], [-1, 2]]`

So:
\[ P^{-1} = \begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix} \]

Thus, the diagonalization of A is:
\[ A = P \cdot D \cdot P^{-1} \]
\[ A = \begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix} \cdot \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix} \cdot \begin{bmatrix} 1 & -1 \\ -1 & 2 \end{bmatrix} \]

#### 6.2 Diagonalization of B:
From the previous section, we know the eigenvalues of B are \( \lambda_1 = 3 \) and \( \lambda_2 = -1 \), and the corresponding eigenvectors are:
- \( v_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix} \)
- \( v_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix} \)

Thus, matrix P is:
\[ P = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix} \]

The diagonal matrix D is:
\[ D = \begin{bmatrix} 3 & 0 \\ 0 & -1 \end{bmatrix} \]

The inverse of P is:
\[ P^{-1} = \frac{1}{\text{det}(P)} \cdot \text{adj}(P) \]

- det(P) = (1 * 1) - (1 * -1) = 2
- adj(P) = `[[1, 1], [-1, 1]]`

So:
\[ P^{-1} = \frac{1}{2} \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix} \]

Thus, the diagonalization of B is:
\[ B = P \cdot D \cdot P^{-1} \]
\[ B = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix} \cdot \begin{bmatrix} 3 & 0 \\ 0 & -1 \end{bmatrix} \cdot \frac{1}{2} \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix} \]

#### 6.3 Diagonalization of C:
Matrix C is already diagonal, as it is a diagonal matrix:
\[ C = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} \]

Thus, C is already in the form \( P \cdot D \cdot P^{-1} \) where:
- P is the identity matrix:
  \[ P = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \]
- D is the diagonal matrix:
  \[ D = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} \]
- \( P^{-1} = P \), as the inverse of the identity matrix is itself.

So, the diagonalization of C is simply:
\[ C = P \cdot D \cdot P^{-1} \]
where \( P = I \) and \( D = C \).







