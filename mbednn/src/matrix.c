/**********************************************************************************/
/* Copyright (c) 2025 Markus Geiger                                               */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files (the "Software"), to deal  */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#include <stdio.h>

#include "alloc.h"
#include "random.h"
#include "matrix.h"

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_create
**
**     brief    Create a new (empty) matrix.
**
**     params   rows: number of rows in the matrix
**              cols: number of columns in the matrix
**     return   pointer to the created matrix
*/
/*---------------------------------------------------------------------------*/
matrix_t *matrix_create(uint32_t rows, uint32_t cols)
{
    matrix_t *m;

    m = alloc_aligned_malloc(sizeof(matrix_t), sizeof(uintptr_t));
    if (m != NULL)
    {
        m->values = alloc_aligned_malloc(rows * cols * sizeof(real), sizeof(real));
        if (m->values == NULL)
        {
            alloc_aligned_free(m);
            m = NULL;
        }
        else
        {
            m->rows = rows;
            m->cols = cols;
        }
    }

    return m;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_create_from_array_soft
**
**     brief    Create a new matrix from an existing array (zero copy) of real numbers.
**
**     params   rows: number of rows in the matrix
**              cols: number of columns in the matrix
**              vals: flattened array of real numbers of the matrix
**     return   pointer to the created matrix
*/
/*---------------------------------------------------------------------------*/
matrix_t* matrix_create_from_array_soft(uint32_t rows, uint32_t cols, real *vals)
{
    matrix_t* m;

    m = alloc_aligned_malloc(sizeof(matrix_t), sizeof(uintptr_t));
    if (m != NULL)
    {
        m->values = vals;
        if (m->values == NULL)
        {
            alloc_aligned_free(m);
            m = NULL;
        }
        else
        {
            m->rows = rows;
            m->cols = cols;
        }
    }

    return m;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_create_from_array_hard
**
**     brief    Create a new matrix from an array (hard copy) of real numbers.
**
**     params   rows: number of rows in the matrix
**              cols: number of columns in the matrix
**              vals: flattened array of real numbers filled into the matrix
**     return   pointer to the created matrix
*/
/*---------------------------------------------------------------------------*/
matrix_t* matrix_create_from_array_hard(uint32_t rows, uint32_t cols, real *vals)
{
    matrix_t *m = NULL;
    uint32_t cells;

    if (vals != NULL)
    {
        m = matrix_create(rows, cols);
        if (m != NULL)
        {
            cells = m->rows * m->cols;

            for (uint32_t i = 0ul; i < cells; i++)
            {
                m->values[i] = vals[i];
            }
        }
    }

    return m;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_create_zeros
**
**     brief    Create a new matrix initializing all cells with zero.
**
**     params   rows: number of rows in the matrix
**              cols: number of columns in the matrix
**     return   pointer to the created matrix
*/
/*---------------------------------------------------------------------------*/
matrix_t* matrix_create_zeros(uint32_t rows, uint32_t cols)
{
    matrix_t *m;

    m = matrix_create(rows, cols);
    if (m != NULL)
    {
        matrix_fill_value(m, REAL_C(0.0));
    }

    return m;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_free_soft
**
**     brief    Free a matrix memory, but not the values.
**
**     params   m: pointer to the created matrix
**     return   void
*/
/*---------------------------------------------------------------------------*/
void matrix_free_soft(matrix_t *m)
{
    if (m != NULL)
    {
        m->rows = 0ul;
        m->cols = 0ul;
        alloc_aligned_free(m);
    }
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_free_hard
**
**     brief    Free a complete matrix memory.
**
**     params   m: pointer to the created matrix
**     return   void
*/
/*---------------------------------------------------------------------------*/
void matrix_free_hard(matrix_t *m)
{
    if (m != NULL)
    {
        m->rows = 0ul;
        m->cols = 0ul;
        alloc_aligned_free(m->values);
        alloc_aligned_free(m);
    }
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_fill_value
**
**     brief    Fill a matrix with a given value.
**
**     params   m: pointer to a matrix
**              val: the value to be filled in all cells of the matrix
**     return   void
*/
/*---------------------------------------------------------------------------*/
void matrix_fill_value(matrix_t *m, real val)
{
    uint32_t cells;

    if (m != NULL)
    {
        cells = m->rows * m->cols;

        for (uint32_t i = 0ul; i < cells; i++)
        {
            m->values[i] = val;
        }
    }
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_fill_random_uniform
**
**     brief    Fill a matrix with uniform distributed random values.
**
**     params   m: pointer to a matrix
**              min: smallest boundary random value
**              max: biggest boundary random value
**     return   void
*/
/*---------------------------------------------------------------------------*/
void matrix_fill_random_uniform(matrix_t *m, real min, real max)
{
    uint32_t cells;

    if (m != NULL)
    {
        cells = m->rows * m->cols;

        for (uint32_t i = 0ul; i < cells; i++)
        {
            m->values[i] = random_uniform_real(min, max);
        }
    }
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_fill_random_gaussian
**
**     brief    Fill a matrix with normal (gaussian) distributed random values.
**
**     params   m: pointer to a matrix
**              mean: mean
**              stddev: standard deviation
**              min: smallest boundary random value
**              max: biggest boundary random value
**     return   void
*/
/*---------------------------------------------------------------------------*/
void matrix_fill_random_gaussian(matrix_t *m, real mean, real stddev, real min, real max)
{
    uint32_t cells;

    if (m != NULL)
    {
        cells = m->rows * m->cols;

        for (uint32_t i = 0ul; i < cells; i++)
        {
            m->values[i] = random_gaussian_real(mean, stddev, min, max);
        }
    }
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_add_scalar
**
**     brief    Add scalar to a matrix.
**
**     params   m: pointer to a matrix
**              val: the value to be added to each cell of the matrix
**     return   pointer to the matrix
*/
/*---------------------------------------------------------------------------*/
matrix_t* matrix_add_scalar(matrix_t *m, real val)
{
    uint32_t cells;

    if (m != NULL)
    {
        cells = m->rows * m->cols;

        for (uint32_t i = 0ul; i < cells; i++)
        {
            m->values[i] += val;
        }
    }

    return m;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_ax_add_by
**
**     brief    Add two matrices, each multiplied by a scalar.
**              y = alpha * x + beta * y
**
**     params   alpha: scalar of matrix x
**              x: pointer to a matrix x
**              beta: scalar of matrix y
**              y: pointer to a matrix y
**     return   pointer to the matrix y
*/
/*---------------------------------------------------------------------------*/
matrix_t* matrix_ax_add_by(real alpha, matrix_t *x, real beta, matrix_t *y)
{
    matrix_t *m = NULL;
    uint32_t cells;

    if ((x != NULL) && (y != NULL))
    {
        if ((x->rows == y->rows) && (x->cols == y->cols))
        {
            cells = x->rows * x->cols;

            if ((alpha == REAL_C(1.0)) && (beta == REAL_C(1.0)))
            {
                for (uint32_t i = 0ul; i < cells; i++)
                {
                    y->values[i] += x->values[i];
                }
            }
            else if (alpha == REAL_C(1.0))
            {
                for (uint32_t i = 0ul; i < cells; i++)
                {
                    y->values[i] = x->values[i] + beta * y->values[i];
                }
            }
            else if (beta == REAL_C(1.0))
            {
                for (uint32_t i = 0ul; i < cells; i++)
                {
                    y->values[i] += alpha * x->values[i];
                }
            }
            else
            {
                for (uint32_t i = 0ul; i < cells; i++)
                {
                    y->values[i] = alpha * x->values[i] + beta * y->values[i];
                }
            }
            m = y;
        }
    }

    return m;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_scalar_product
**
**     brief    Multiply a matrix by a scalar.
**              m = alpha * m
**
**     params   alpha: scalar of matrix m
**              m: pointer to a matrix m
**     return   pointer to the matrix m
*/
/*---------------------------------------------------------------------------*/
matrix_t* matrix_scalar_product(matrix_t *m, real alpha)
{
    uint32_t cells;

    if (m != NULL)
    {
        cells = m->rows * m->cols;

        for (uint32_t i = 0ul; i < cells; i++)
        {
            m->values[i] *= alpha;
        }
    }

    return m;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_hadamard_product
**
**     brief    Multiply two matrices elementwise (Hadamard product).
**              a = a * b
**
**     params   a: pointer to a matrix a
**              b: pointer to a matrix b
**     return   pointer to the matrix a
*/
/*---------------------------------------------------------------------------*/
matrix_t* matrix_hadamard_product(matrix_t *a, matrix_t *b)
{
    matrix_t *m = NULL;
    uint32_t cells;

    if ((a != NULL) && (b != NULL))
    {
        if ((a->rows == b->rows) && (a->cols == b->cols))
        {
            cells = a->rows * a->cols;

            for (uint32_t i = 0ul; i < cells; i++)
            {
                a->values[i] *= b->values[i];
            }
            m = a;
        }
    }

    return m;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_matrix_vector_product
**
**     brief    Multiply a matrix with a vector (matrix-vector product).
**              d = alpha * a.v + beta * d
**                                          [a11 a12]
**     example  (alpha = 1.0, beta = 0.0):  [a21 a22] . [v1 v2] = [a11v1+a12v2 a21v1+a22v2 a31v1+a32v2]
**                                          [a31 a32]
**
**     params   alpha: scalar of matrix a
**              a: pointer to the matrix a
**              beta: scalar of matrix d
**              v: pointer to the vector v
**              d: pointer to the matrix d
**     return   pointer to the matrix d
*/
/*---------------------------------------------------------------------------*/
matrix_t* matrix_matrix_vector_product(real alpha, matrix_t *a, real beta, matrix_t *v, matrix_t *d)
{
    matrix_t *m = NULL;
    real sum;

    if ((a != NULL) && (v != NULL) && (d != NULL))
    {
        if ((v->rows == 1ul) && (a->cols == v->cols) && (d->rows == 1ul))
        {
            for (uint32_t a_row = 0ul; a_row < a->rows; a_row++)
            {
                sum = REAL_C(0.0);
                for (uint32_t a_col = 0ul; a_col < a->cols; a_col++)
                {
                    sum += alpha * a->values[a_row * a->cols + a_col] * v->values[a_col];
                }
                d->values[a_row] = beta * d->values[a_row] + sum;
            }
            m = d;
        }
    }

    return m;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_matrix_vector_transposed_product
**
**     brief    Multiply a transposed matrix with a transposed vector 
**              (transposed matrix-vector product).
**              d = alpha * AT.vT + beta * dT = alpha * (v.A)T + beta * d
**                                                             ([v1])T    (             [a11 a12])T
**     example  (alpha = 1.0, beta = 0.0):  ([a11 a21 a31])T . ([v2])   = ([v1 v2 v3] . [a21 a22])  = [a11v1+a21v2+a31v3 a12v1+a22v2+a32v3]
**                                          ([a12 a22 a32])    ([v3])     (             [a31 a32])
**
**     params   alpha: scalar of matrix a
**              a: pointer to the matrix a
**              beta: scalar of matrix d
**              v: pointer to the vector v
**              d: pointer to the matrix d
**     return   pointer to the matrix d
*/
/*---------------------------------------------------------------------------*/
matrix_t* matrix_matrix_vector_transposed_product(real alpha, matrix_t *a, real beta, matrix_t *v, matrix_t *d)
{
    matrix_t *m = NULL;
    real sum;

    if ((a != NULL) && (v != NULL) && (d != NULL))
    {
        if ((v->rows == 1ul) && (a->rows == v->cols) && (d->rows == 1ul))
        {
            for (uint32_t a_col = 0ul; a_col < a->cols; a_col++)
            {
                sum = REAL_C(0.0);
                for (uint32_t a_row = 0ul; a_row < a->rows; a_row++)
                {
                    sum += alpha * a->values[a_row * a->cols + a_col] * v->values[a_row];
                }
                d->values[a_col] = beta * d->values[a_col] + sum;
            }
            m = d;
        }
    }

    return m;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_vector_outer_product
**
**     brief    Multiply a vector with another vector (vector outer product).
**              d += alpha * u x v
**                                                     [u1v1 u1v2]
**     example  (alpha = 1.0):  [u1 u2 u3] x [v1 v2] = [u2v1 u2v2]
**                                                     [u3v1 u3v2]
**
**     params   alpha: scalar
**              u: pointer to the vector u
**              v: pointer to the vector v
**              d: pointer to the matrix d
**     return   pointer to the matrix d
*/
/*---------------------------------------------------------------------------*/
matrix_t* matrix_vector_outer_product(real alpha, matrix_t *u, matrix_t *v, matrix_t *d)
{
    matrix_t *m = NULL;

    if ((u != NULL) && (v != NULL) && (d != NULL))
    {
        if ((u->rows == 1ul) && (v->rows == 1ul) && (u->cols == d->rows) && (v->cols == d->cols))
        {
            for (uint32_t u_col = 0ul; u_col < u->cols; u_col++)
            {
                for (uint32_t v_col = 0ul; v_col < v->cols; v_col++)
                {
                    d->values[u_col * v->cols + v_col] += alpha * u->values[u_col] * v->values[v_col];
                }
            }
            m = d;
        }
    }

    return m;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: matrix_print
**
**     brief    Print a matrix (for debugging).
**
**     params   m: pointer to a matrix m
**     return   void
*/
/*---------------------------------------------------------------------------*/
void matrix_print(matrix_t *m)
{
    if (m != NULL)
    {
        MATRIX_PRINTF("[");
        for (uint32_t row = 0ul; row < m->rows; row++)
        {
            MATRIX_PRINTF("[");
            for (uint32_t col = 0ul; col < m->cols; col++)
            {
                if (col != 0ul)
                {
                    MATRIX_PRINTF(",");
                }
                MATRIX_PRINTF(" %f", m->values[row * m->cols + col]);
            }
            MATRIX_PRINTF("]");
            if ((row + 1ul) != (m->rows))
            {
                MATRIX_PRINTF(", ");
            }
        }
        MATRIX_PRINTF("]\n");
    }
}
