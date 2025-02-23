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

#ifndef __TYPES_H
#define __TYPES_H

#include <float.h>

#include "config.h"

#ifdef MBEDNN_HIGH_PRECISION
typedef double real;
#define REAL_C(a)          a
#define REAL_COMP_EPSILON  DBL_EPSILON
#define REAL_DIV_EPSILON   (1.0 / DBL_MAX)
#define REAL_LOG_EPSILON   DBL_MIN
#define REAL_SQRT_EPSILON  1e-317
#define REAL_EXP(a)        exp(a)
#define REAL_FMAX(a, b)    fmax(a, b)
#define REAL_TANH(a)       tanh(a)
#define REAL_FABS(a)       fabs(a)
#define REAL_LOG(a)        log(a)
#define REAL_SQRT(a)       sqrt(a)
#define REAL_POW(a, b)     pow(a, b)
#else
typedef float real;
#define REAL_C(a)          a##f
#define REAL_COMP_EPSILON  FLT_EPSILON
#define REAL_DIV_EPSILON   (1.0f / FLT_MAX)
#define REAL_LOG_EPSILON   FLT_MIN
#define REAL_SQRT_EPSILON  1e-17f
#define REAL_EXP(a)        expf(a)
#define REAL_FMAX(a, b)    fmaxf(a, b)
#define REAL_TANH(a)       tanhf(a)
#define REAL_FABS(a)       fabsf(a)
#define REAL_LOG(a)        logf(a)
#define REAL_SQRT(a)       sqrtf(a)
#define REAL_POW(a, b)     powf(a, b)
#endif

#endif
