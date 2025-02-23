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

#include "..\\..\\mbednn\\src\\mbednn.h"

#define SAMPLES             4ul
#define INPUTS_PER_SAMPLE   2ul
#define OUTPUTS_PER_SAMPLE  1ul

real data[SAMPLES * INPUTS_PER_SAMPLE] = {
    REAL_C(0.0), REAL_C(0.0),
    REAL_C(0.0), REAL_C(1.0),
    REAL_C(1.0), REAL_C(0.0),
    REAL_C(1.0), REAL_C(1.0)
};

real result[SAMPLES * OUTPUTS_PER_SAMPLE] = {
    REAL_C(0.0),
    REAL_C(1.0),
    REAL_C(1.0),
    REAL_C(0.0)
};

int main(int argc, char* argv[])
{
    real outputs[OUTPUTS_PER_SAMPLE];
    real inputs[INPUTS_PER_SAMPLE];

    // setup training tensors
    matrix_t *x_train = matrix_create_from_array_soft(SAMPLES, INPUTS_PER_SAMPLE, data);
    matrix_t *y_train = matrix_create_from_array_soft(SAMPLES, OUTPUTS_PER_SAMPLE, result);

    // create a new empty fully connected neural network
    mbednn_t *mbednn = mbednn_create();

    // define network layers
    mbednn_add_layer_input(mbednn, INPUTS_PER_SAMPLE);
    mbednn_add_layer_dense(mbednn, 2ul, MBEDNN_ACTIVATION_SIGMOID, MBEDNN_RANDOM_UNIFORM);
    mbednn_add_layer_output(mbednn, OUTPUTS_PER_SAMPLE, MBEDNN_ACTIVATION_SIGMOID, MBEDNN_RANDOM_UNIFORM);

    // show network summary
    mbednn_summary(mbednn);

    // network settings
    mbednn_compile(mbednn, MBEDNN_OPTIMIZER_DEFAULT, MBEDNN_LOSS_MSE, REAL_C(0.01), REAL_C(0.001));

    // train the network
    mbednn_fit(mbednn, x_train, y_train, 100000ul, 32ul);

    // make a prediction using the trained network
    inputs[0u] = REAL_C(0.0);
    inputs[1u] = REAL_C(0.0);
    mbednn_predict(mbednn, inputs, outputs);
    mbednn_print_outputs(mbednn);

    inputs[0u] = REAL_C(0.0);
    inputs[1u] = REAL_C(1.0);
    mbednn_predict(mbednn, inputs, outputs);
    mbednn_print_outputs(mbednn);

    inputs[0u] = REAL_C(1.0);
    inputs[1u] = REAL_C(0.0);
    mbednn_predict(mbednn, inputs, outputs);
    mbednn_print_outputs(mbednn);

    inputs[0u] = REAL_C(1.0);
    inputs[1u] = REAL_C(1.0);
    mbednn_predict(mbednn, inputs, outputs);
    mbednn_print_outputs(mbednn);

    mbednn_free(mbednn);
    matrix_free_soft(x_train);
    matrix_free_soft(y_train);

    return 0;
}
