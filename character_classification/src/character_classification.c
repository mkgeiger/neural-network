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
#include <stdlib.h>

#include "..\\..\\mbednn\\src\\mbednn.h"

// use convolution neural network
#define USE_CNN

#ifndef USE_CNN
#define ROWS                7ul
#define COLS                5ul
#else
#define ROWS               16ul
#define COLS               12ul
#endif

#define SAMPLES            10ul            // 10 numbers
#define INPUTS_PER_SAMPLE  (ROWS * COLS)
#define CLASSES            SAMPLES

#ifndef USE_CNN
real data[SAMPLES * INPUTS_PER_SAMPLE] = {
    // character '0'
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),

    // character '1'
    REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0), REAL_C(0.0),

    // character '2'
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),

    // character '3'
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),

    // character '4'
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0),
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0),

    // character '5'
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),

    // character '6'
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),

    // character '7'
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),

    // character '8'
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),

    // character '9'
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0)
};
#else
real data[SAMPLES * INPUTS_PER_SAMPLE] = {
    // character '0'
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),

    // character '1'
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),

    // character '2'
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),

    // character '3'
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),

    // character '4'
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0),

    // character '5'
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),

    // character '6'
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),

    // character '7'
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),

    // character '8'
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),

    // character '9'
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(1.0), REAL_C(1.0), REAL_C(0.5),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5), REAL_C(0.5),
};
#endif

real result[SAMPLES * CLASSES] = {
    REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(0.0), REAL_C(1.0)
};

// display a character from flat data
static void print_character(real* data)
{
    char c;

    printf("Input vector is:\n");
    for (uint32_t row = 0ul; row < ROWS; row++)
    {
        for (uint32_t col = 0ul; col < COLS; col++)
        {
            if (data[row * COLS + col] == REAL_C(1.0))
            {
                c = '+';
            }
            else
            {
                c = ' ';
            }
            printf("%c", c);
        }
        printf("\n");
    }
}

// add some noise to a character
static void add_noise_to_character(real* data, uint32_t size, uint32_t amount)
{
    uint32_t index;

    for (uint32_t i = 0ul; i < amount; i++)
    {
        index = (uint32_t)(rand() % size);
        if (data[index] == REAL_C(1.0))
        {
            data[index] = REAL_C(0.0);
        }
        else
        {
            data[index] = REAL_C(1.0);
        }
    }
}

int main(int argc, char* argv[])
{
    uint32_t correct = 0ul;
    real outputs[CLASSES];
    uint32_t pred_class;
    uint32_t act_class;
    real accuracy;
    real loss;
    uint32_t conv2d_layer[2ul];

    // setup training matrices
    matrix_t *x_train = matrix_create_from_array_soft(SAMPLES, INPUTS_PER_SAMPLE, data);
    matrix_t *y_train = matrix_create_from_array_soft(SAMPLES, CLASSES, result);

    // create a new empty fully connected neural network
    mbednn_t *mbednn = mbednn_create();

    // define network layers
#ifdef USE_CNN
    (void)mbednn_add_layer_input_2d(mbednn, ROWS, COLS, 3ul);
    conv2d_layer[0ul] = mbednn_add_layer_conv_2d(mbednn, 8ul, 3ul, 3ul, 1ul, MBEDNN_ACTIVATION_RELU);
    (void)mbednn_add_layer_maxpooling_2d(mbednn);
    conv2d_layer[1ul] = mbednn_add_layer_conv_2d(mbednn, 16ul, 3ul, 3ul, 1ul, MBEDNN_ACTIVATION_RELU);
    (void)mbednn_add_layer_maxpooling_2d(mbednn);
    (void)mbednn_add_layer_flatten_2d(mbednn);
#else
    (void)mbednn_add_layer_input(mbednn, INPUTS_PER_SAMPLE);
#endif
    (void)mbednn_add_layer_dense(mbednn, 48ul, MBEDNN_ACTIVATION_SIGMOID, MBEDNN_RANDOM_UNIFORM);
    (void)mbednn_add_layer_output(mbednn, CLASSES, MBEDNN_ACTIVATION_SIGMOID, MBEDNN_RANDOM_UNIFORM);

    // show network summary
    mbednn_summary(mbednn);

    // define network hyper parameters
    mbednn_compile(mbednn, MBEDNN_OPTIMIZER_DEFAULT, MBEDNN_LOSS_DEFAULT, REAL_C(0.01), REAL_C(0.0001));

    // train the network
    loss = mbednn_fit(mbednn, x_train, y_train, 10000ul, 32ul);
    printf("Training loss: %f\n", loss);

    // get the accuracy of the trained network
    accuracy = mbednn_get_accuracy(mbednn, x_train, y_train);
    printf("Training accuracy: %f\n", accuracy * REAL_C(100.0));

    // test the network
    printf("Start the prediction of test samples ...\n\n");
    for (uint32_t i = 0ul; i < SAMPLES; i++)
    {
#ifdef USE_CNN
        // modify sample with 12 errors
        add_noise_to_character(&data[i * INPUTS_PER_SAMPLE], INPUTS_PER_SAMPLE, 12ul);
#else
        // modify sample with 3 errors
        add_noise_to_character(&data[i * INPUTS_PER_SAMPLE], INPUTS_PER_SAMPLE, 3ul);
#endif

        mbednn_predict(mbednn, &data[i * INPUTS_PER_SAMPLE], outputs);

        print_character(&data[i * INPUTS_PER_SAMPLE]);

        pred_class = mbednn_class_predict(outputs, CLASSES);
        printf("Predicted class is: %ld\n", pred_class % CLASSES);

        act_class = mbednn_class_predict(&y_train->values[i * CLASSES], CLASSES);
        printf("Actual class is: %ld\n\n", act_class % CLASSES);

        if (pred_class == act_class)
        {
            correct++;
        }
    }
    printf("Test accuracy: %f\n", (real)correct * REAL_C(10.0));

    mbednn_free(mbednn);
    matrix_free_soft(x_train);
    matrix_free_soft(y_train);

    return 0;
}
