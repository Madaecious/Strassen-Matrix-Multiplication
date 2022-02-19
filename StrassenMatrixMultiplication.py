###############################################################################
# Implementation of Strassen Matrix Multiplication
# Mark Barros - BID 013884117
# CS3310 - Design and Analysis of Algorithms
# Cal Poly Pomona: Spring 2021
###############################################################################

# These are imported modules. -------------------------------------------------
import numpy as np
import time
#------------------------------------------------------------------------------

# This generates an nxn matrix. -----------------------------------------------

def matrixGenerator(n):
        return np.random.randint(100,999, size=(n, n))

# This splits the given nxn matrix into quarters. It returns a tuple
# containing four n/2 x n/2 matrices corresponding to a, b, c, d.

def split(matrix):   
    row, col = matrix.shape
    row2, col2 = row//2, col//2
    return matrix[:row2, :col2], matrix[:row2, col2:], \
        matrix[row2:, :col2], matrix[row2:, col2:] 

# This computes the matrix product using Strassen's ---------------------------
# divide and conquer approach.

def strassenMultiply(matrixA, matrixB): 
  
    # The base case is when the matrix size is 1x1. 
    if len(matrixA) == 1: 
        return matrixA * matrixB
  
    # This recursively splits the matrices into quadrants 
    # until the base case is reached. 
    a, b, c, d = split(matrixA) 
    e, f, g, h = split(matrixB) 
  
    # This recursively computes the seven products. 
    p1 = strassenMultiply(a, f - h)   
    p2 = strassenMultiply(a + b, h)         
    p3 = strassenMultiply(c + d, e)         
    p4 = strassenMultiply(d, g - e)
    p5 = strassenMultiply(a + d, e + h)         
    p6 = strassenMultiply(b - d, g + h)   
    p7 = strassenMultiply(a - c, e + f)   
  
    # This computes the values of the four quadrants of the final matrix c.
    c11 = p5 + p4 - p2 + p6   
    c12 = p1 + p2            
    c21 = p3 + p4             
    c22 = p1 + p5 - p3 - p7   
  
    # This combines the four quadrants into a single matrix by stacking
    # them horizontally and vertically.
    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))  
  
    return c

# This is the driver code. ----------------------------------------------------

if __name__ == '__main__':

    # This is the output header.
    print("------------------------------------------------------")
    print("Mark Barros")
    print("Implementation of Strassen Matrix Multiplication")
    print("CS3310 - Design and Analysis of Algorithms")
    print("Cal Poly Pomona: Spring 2021")
    print("------------------------------------------------------")


    product = 0 
    n = 2               # n is both dimensions of each 2D array (matrix).

    # This loop provides the option of repeatedly performing matrix
    # multiplication (doubling the size of the matrices each time).
    while n <= 2:

        # This generates two randomly populated matrices.
        MatrixA = matrixGenerator(n)
        MatrixB = matrixGenerator(n)

        # This outputs to the console the matrices.
        print("Matrix A:")
        for row in MatrixA:
            print(*row)

        print("\nMatrix B:")
        for row in MatrixB:
            print(*row)

        # This performs a strassenMultiply and times the operation. -----------
        start = time.perf_counter_ns()
        product = strassenMultiply(MatrixA, MatrixB)
        finish = time.perf_counter_ns()

        # This outputs to the console the product.
        print("\nProduct:")
        for row in product:
            print(*row)

        # This calculates the elapsed time in seconds.
        period = ((finish - start) * (10**-9))

        # This outputs to the screen the length of the matrix sides in each
        # instance and the corresponding time it took to multiply them.
        print("------------------------------------------------------")
        print("Exectution Time: For n = ", f'{n:3,}', " t = ", f'{period:.3}')
        print("------------------------------------------------------")

        # This doubles the sides of the matrices to be multiplied
        # after each iteration.
        n = 2 * n
###############################################################################