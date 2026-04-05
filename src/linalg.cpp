#include "../include/linalg.hpp"
#include <iostream>
#include <vector>

using namespace std;

void multiply_mv_row_major(const double* matrix, int rows, int cols, const double* vector, double* result)
{
    for (int i = 0; i < rows; ++i)
    {
        result[i] = 0;
        for (int j = 0; j < cols; ++j)
        {
            result[i] += matrix[i * cols + j] * vector[j];

        };
    };
};


void multiply_mv_col_major(const double* matrix, int rows, int cols, const double* vector, double* result)
{
    for (int i = 0; i < rows; ++i)
    {
        result[i] = 0;
        for (int j = 0; j < cols; ++j)
        {
            result[i] += matrix[j * rows + i] * vector[j];

        };
    };
};

void allocate_2d_matrix(double*** matrix, int rows, int cols){
    *matrix = (double**)malloc(sizeof(double) * rows * cols);
    
    double (*B)[cols] = (double(*)[cols])(*matrix);
    
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            B[i][j] = i+j;
        }
    }
}

void multiply_matrices(const double **A_, const double **B_, double **C_, int rows, int inner, int cols){
    
    
    double (*A)[inner] = (double(*)[inner])A_;
    double (*B)[cols] = (double(*)[cols])B_;
    double (*C)[cols] = (double(*)[cols])C_;
    
    int sum;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            sum = 0;
            for(int k=0;k<inner;k++){
                sum += A[i][k]*B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void multiply_mm_naive(const double* matrixA, int rowsA, int colsA, const double* matrixB, int rowsB, int colsB, double* result){
    if(colsA!=rowsB){
        cout << "Wrong dimensions for multiplication" << endl;
        return;
    }
    
    if((matrixA == nullptr) || (matrixB == nullptr)){
        cout << "either or both of matrices have nullptr" << endl;
        return;
    }
    
    multiply_matrices((const double**)matrixA, (const double**)matrixB, (double**)result, rowsA, colsA, colsB);
    
}

void allocate_2d_matrix_column_major(double*** matrix, int rows, int cols){
    *matrix = (double**)malloc(sizeof(double) * rows * cols);
    
    double (*B)[rows] = (double(*)[rows])(*matrix);
    
    for(int i=0;i<cols;i++){
        for(int j=0;j<rows;j++){
            B[i][j] = i+j;
        }
    }
}

void multiply_matrices_tranposed_b(const double **A_, const double **B_, double **C_, int rows, int inner, int cols){
    
    
    double (*A)[inner] = (double(*)[inner])A_;
    double (*B)[inner] = (double(*)[inner])B_;
    double (*C)[cols] = (double(*)[cols])C_;
    
    int sum;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            sum = 0;
            for(int k=0;k<inner;k++){
                sum += A[i][k]*B[j][k];
            }
            C[i][j] = sum;
        }
    }
}


void multiply_mm_transposed_b(const double* matrixA, int rowsA, int colsA, const double* matrixB_transposed, int rowsB, int colsB, double* result){
    if(colsA!=rowsB){
        cout << "Wrong dimensions for multiplication" << endl;
        return;
    }
    
    if((matrixA == nullptr) || (matrixB_transposed == nullptr)){
        cout << "either or both of matrices have nullptr" << endl;
        return;
    }
    
    multiply_matrices_tranposed_b((const double**)matrixA, (const double**)matrixB_transposed, (double**)result, rowsA, colsA, colsB);
    
}


void printMatrix(double **A_, int rows, int cols){
    double (*A)[cols] = (double(*)[cols])A_;
    
    cout << endl;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            cout << A[i][j] << " ";
        }
        cout << endl;
    }
    cout << "finish printing \n\n";
}

void freeMatrix(double ***A_){
    free(*A_);
    *A_ = nullptr;
}