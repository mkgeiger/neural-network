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

#ifndef __CONFIG_H
#define __CONFIG_H

#undef  MBEDNN_HIGH_PRECISION                            // use double precision instead of single precision real type

#define MBEDNN_USE_TRAINING                              // compile with training code (#undef to save memory when backward path is not required)

#define MBEDNN_PRINTF                     printf         // neural network debug printf

#define MBEDNN_DEFAULT_EPSILON            REAL_C(0.001)  // epsilon <= 0.1% is default

#define MBEDNN_DEFAULT_LEARNING_RATE      REAL_C(0.03)   // base learning rate

#define MBEDNN_DEFAULT_LEARNING_DECAY     REAL_C(0.9995) // decay rate for learning rate

#define MBEDNN_DEFAULT_LEARN_ADD          REAL_C(0.005)  // adaptive learning rate factor

#define MBEDNN_DEFAULT_LEARN_SUB          REAL_C(0.75)   // adaptive learning rate factor

#define MBEDNN_DEFAULT_MSE_AVG            4ul            // number of prior MSE's to average

#define MBEDNN_DEFAULT_EPOCHS             10000ul        // default epoch size

#define MBEDNN_DEFAULT_BATCH_SIZE         32ul           // default batch size

#endif
