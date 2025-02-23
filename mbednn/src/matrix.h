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

#ifndef __MATRIX_H
#define __MATRIX_H

#include <stdint.h>

#include "types.h"

// Matrix Debug printf
#define MATRIX_PRINTF   printf

// Matrix Structure
typedef struct
{
    uint32_t rows;
    uint32_t cols;
    real     *values;
} matrix_t;

// Matrix Create/Free
matrix_t* matrix_create(uint32_t rows, uint32_t cols);
matrix_t* matrix_create_from_array_soft(uint32_t rows, uint32_t cols, real *vals);
matrix_t* matrix_create_from_array_hard(uint32_t rows, uint32_t cols, real *vals);
matrix_t* matrix_create_zeros(uint32_t rows, uint32_t cols);
void      matrix_free_soft(matrix_t *m);
void      matrix_free_hard(matrix_t *m);

// Matrix Fill
void      matrix_fill_value(matrix_t *m, real val);
void      matrix_fill_random_uniform(matrix_t *m, real min, real max);
void      matrix_fill_random_gaussian(matrix_t *m, real mean, real stddev, real min, real max);

// Matrix Math
matrix_t* matrix_add_scalar(matrix_t *m, real val);
matrix_t* matrix_ax_add_by(real alpha, matrix_t *x, real beta, matrix_t *y);
matrix_t* matrix_scalar_product(matrix_t *m, real val);
matrix_t* matrix_hadamard_product(matrix_t *a, matrix_t *b);
matrix_t* matrix_matrix_vector_product(real alpha, matrix_t *a, real beta, matrix_t *v, matrix_t *d);
matrix_t* matrix_matrix_vector_transposed_product(real alpha, matrix_t *a, real beta, matrix_t *v, matrix_t *d);
matrix_t* matrix_vector_outer_product(real alpha, matrix_t *u, matrix_t *v, matrix_t *d);

// Matrix Debug
void      matrix_print(matrix_t *m);

#endif
