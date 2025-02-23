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

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "config.h"
#include "alloc.h"
#include "random.h"
#include "matrix.h"
#include "io.h"
#include "mbednn.h"

// Compute the null activation
static real mbednn_null_activation(real x)
{
    return x;
}

// Compute the null derivation
static real mbednn_null_derivation(real x)
{
    return REAL_C(0.0);
}

// Compute the sigmoid activation
static real mbednn_sigmoid_activation(real x)
{
    return (REAL_C(1.0) / (REAL_C(1.0) + REAL_EXP(-x)));
}

// Compute the sigmoid derivation
static real mbednn_sigmoid_derivation(real x)
{
    return (mbednn_sigmoid_activation(x) * (REAL_C(1.0) - mbednn_sigmoid_activation(x)));
}

// Compute the ReLU activation
static real mbednn_relu_activation(real x)
{
    return REAL_FMAX(REAL_C(0.0), x);
}

// Compute the ReLU derivation (Heaviside function)
static real mbednn_relu_derivation(real x)
{
    return (x > REAL_C(0.0) ? REAL_C(1.0) : REAL_C(0.0));
}

// Compute the leaky ReLU activation
static real mbednn_leaky_relu_activation(real x)
{
    return REAL_FMAX(REAL_C(0.01) * x, x);
}

// Compute the leaky ReLU derivation
static real mbednn_leaky_relu_derivation(real x)
{
    return (x > REAL_C(0.0) ? REAL_C(1.0) : REAL_C(0.01));
}

// Compute the tanh activation
static real mbednn_tangens_hyperbolicus_activation(real x)
{
    return REAL_TANH(x);
}

// Compute the tanh derivation
static real mbednn_tangens_hyperbolicus_derivation(real x)
{
    return (REAL_C(1.0) - REAL_TANH(x) * REAL_TANH(x));
}

// Compute the softsign activation
static real mbednn_softsign_activation(real x)
{
    return (x / (REAL_C(1.0) + REAL_FABS(x)));
}

// Compute the softsign derivation
static real mbednn_softsign_derivation(real x)
{
    return (REAL_C(1.0) / ((REAL_C(1.0) + REAL_FABS(x)) * (REAL_C(1.0) + REAL_FABS(x))));
}

// Apply the softmax function to the output layer's values
static void mbednn_softmax(mbednn_t *mbednn)
{
    real sum = REAL_C(0.0);
    uint32_t output_layer = mbednn->layer_count - 1ul;
    uint32_t node_count = mbednn->layers[output_layer].node_count;

    for (uint32_t node = 0ul; node < node_count; node++)
    {
        sum += REAL_EXP(mbednn->layers[output_layer].outputs[0]->values[node]);
    }

    if (sum == REAL_C(0.0))
    {
        for (uint32_t node = 0ul; node < node_count; node++)
        {
            mbednn->layers[output_layer].outputs[0]->values[node] = REAL_C(0.0);
        }
    }
    else
    {
        for (uint32_t node = 0ul; node < node_count; node++)
        {
            mbednn->layers[output_layer].outputs[0]->values[node] = REAL_EXP(mbednn->layers[output_layer].outputs[0]->values[node]) / sum;
        }
    }
}

#ifdef MBEDNN_USE_TRAINING
// Shuffle the indices (Fisher–Yates shuffle)
static void mbednn_shuffle_indices(uint32_t* input_indices, uint32_t count)
{
    uint32_t value;
    uint16_t j;

    for (uint16_t i = 0u; i < count - 2u; i++)
    {
        j = random_uniform_integer(i, count - 1u);
        value = input_indices[i];
        input_indices[i] = input_indices[j];
        input_indices[j] = value;
    }
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Initialize the weights and biases for each layer in the neural network using the specified random initialization method.
static void mbednn_init_weights(mbednn_t *mbednn)
{
    uint32_t weight_count;
    uint32_t node_count;
    real limit;
    real stddev;
    layer_t *prev_layer;

    if (mbednn != NULL)
    {
        // for all layers except the input layer
        for (uint32_t layer = 1ul; layer < mbednn->layer_count; layer++)
        {
            switch (mbednn->layers[layer].type)
            {
            case MBEDNN_LAYER_CONV_2D:
                // initialize the biases of the filters with random values
                matrix_fill_random_uniform(mbednn->layers[layer].filters_biases, REAL_C(-0.05), REAL_C(0.05));

                for (uint32_t d = 0ul; d < mbednn->layers[layer].depth; d++)
                {
                    for (uint32_t f = 0ul; f < mbednn->layers[layer].filters_count; f++)
                    {
                        // only if not externally configured filter
                        if (mbednn->layers[layer].filters_extern_enable[d][f] == false)
                        {
                            // initialize the filter values with random values
                            matrix_fill_random_uniform(mbednn->layers[layer].filters[d][f], REAL_C(-0.05), REAL_C(0.05));
                        }
                    }
                }
                break;

            case MBEDNN_LAYER_DENSE:
            case MBEDNN_LAYER_OUTPUT:
                prev_layer = &mbednn->layers[layer - 1ul];

                // initialize the biases with random values
                matrix_fill_random_uniform(mbednn->layers[layer].biases, REAL_C(-0.05), REAL_C(0.05));

                // initialize the weights with the related random values
                switch (mbednn->layers[layer].random_type)
                {
                case MBEDNN_RANDOM_UNIFORM:
                    matrix_fill_random_uniform(mbednn->layers[layer].weights, REAL_C(-0.05), REAL_C(0.05));
                    break;

                case MBEDNN_RANDOM_NORMAL:
                    matrix_fill_random_gaussian(mbednn->layers[layer].weights, REAL_C(0.0), REAL_C(0.05), REAL_C(-1.0), REAL_C(1.0));
                    break;

                case MBEDNN_RANDOM_GLOROT_UNIFORM:
                    weight_count = prev_layer->node_count;
                    node_count = mbednn->layers[layer].node_count;
                    limit = REAL_SQRT(REAL_C(6.0) / ((real)(weight_count + node_count)));
                    matrix_fill_random_uniform(mbednn->layers[layer].weights, -limit, limit);
                    break;

                case MBEDNN_RANDOM_GLOROT_NORMAL:
                    weight_count = prev_layer->node_count;
                    node_count = mbednn->layers[layer].node_count;
                    stddev = REAL_SQRT(REAL_C(2.0) / ((real)(weight_count + node_count)));
                    matrix_fill_random_gaussian(mbednn->layers[layer].weights, REAL_C(0.0), stddev, REAL_C(-1.0), REAL_C(1.0));
                    break;

                case MBEDNN_RANDOM_HE_UNIFORM:
                    weight_count = prev_layer->node_count;
                    limit = REAL_SQRT(REAL_C(6.0) / ((real)weight_count));
                    matrix_fill_random_uniform(mbednn->layers[layer].weights, -limit, limit);
                    break;

                case MBEDNN_RANDOM_HE_NORMAL:
                    weight_count = prev_layer->node_count;
                    stddev = REAL_SQRT(REAL_C(2.0) / ((real)weight_count));
                    matrix_fill_random_gaussian(mbednn->layers[layer].weights, REAL_C(0.0), stddev, REAL_C(-1.0), REAL_C(1.0));
                    break;

                case MBEDNN_RANDOM_ONES:
                    matrix_fill_value(mbednn->layers[layer].weights, REAL_C(1.00));
                    break;

                case MBEDNN_RANDOM_ZEROS:
                case MBEDNN_RANDOM_NONE:
                default:
                    matrix_fill_value(mbednn->layers[layer].weights, REAL_C(0.00));
                    break;
                }
                break;

            default:
                break;
            }
        }
    }
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Compute the mean squared error for the output layer
static real mbednn_compute_ms_error(mbednn_t *mbednn, matrix_t *outputs)
{
    real diff;
    real mse = REAL_C(0.0);

    // get the output layer
    layer_t *layer = &mbednn->layers[mbednn->layer_count - 1ul];

    if (layer->type == MBEDNN_LAYER_OUTPUT)
    {
        for (uint32_t node = 0ul; node < layer->node_count; node++)
        {
            diff = outputs->values[node] - layer->outputs[0]->values[node];
            mse += diff * diff;
        }
    }

    return (mse / REAL_C(2.0));
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Compute the categorical cross entropy error for the output layer
static real mbednn_compute_cross_entropy(mbednn_t *mbednn, matrix_t *outputs)
{
    real cross_entropy = REAL_C(0.0);
    real predicted;

    // get the output layer
    layer_t *layer = &mbednn->layers[mbednn->layer_count - 1ul];

    if (layer->type == MBEDNN_LAYER_OUTPUT)
    {
        for (uint32_t node = 0ul; node < layer->node_count; node++)
        {
            predicted = REAL_FMAX(layer->outputs[0]->values[node], REAL_LOG_EPSILON);
            cross_entropy += (outputs->values[node] * REAL_LOG(predicted));
        }
    }

    return -cross_entropy;
}
#endif

static void mbednn_normalize_feature_map(real *feature_map, uint32_t size, real *gamma, real *beta)
{
    real mean = REAL_C(0.0);
    real variance = REAL_C(0.0);

    // compute mean
    for (uint32_t i = 0ul; i < size; i++)
    {
        mean += feature_map[i];
    }
    mean /= (real)size;

    // compute variance
    for (uint32_t i = 0ul; i < size; i++)
    {
        variance += (feature_map[i] - mean) * (feature_map[i] - mean);
    }
    variance /= (real)size;

    // normalize the feature map
    for (uint32_t i = 0ul; i < size; i++)
    {
        variance = REAL_FMAX(variance, REAL_SQRT_EPSILON);
        feature_map[i] = (*gamma) * ((feature_map[i] - mean) / REAL_SQRT(variance)) + (*beta);
    }
}

// Perform convolution on a 2D input layer and store the result in the 2D output layer
static void mbednn_conv_2d_layer(layer_t *prev_layer, layer_t *layer)
{
    uint32_t input_rows = prev_layer->rows;
    uint32_t input_cols = prev_layer->cols;
    uint32_t output_rows = layer->rows;
    uint32_t output_cols = layer->cols;
    uint32_t filter_rows = layer->filters_rows;
    uint32_t filter_cols = layer->filters_cols;
    real output_value;
    real sum;
    real gamma = REAL_C(1.0); // Scaling factor
    real beta = REAL_C(0.0);  // Shifting factor

    for (uint32_t d = 0ul; d < layer->depth; d++)
    {
        for (uint32_t i = 0ul; i < output_rows; i++)
        {
            for (uint32_t j = 0ul; j < output_cols; j++)
            {
                sum = REAL_C(0.0);
                for (uint32_t f = 0ul; f < layer->filters_count; f++)
                {
                    for (uint32_t m = 0ul; m < filter_rows; m++)
                    {
                        for (uint32_t n = 0ul; n < filter_cols; n++)
                        {
                            int32_t input_row = (int32_t)(i * layer->filters_stride) + (int32_t)m - (int32_t)layer->filters_top_padding;
                            int32_t input_col = (int32_t)(j * layer->filters_stride) + (int32_t)n - (int32_t)layer->filters_left_padding;

                            // Check if the filter overlaps the border
                            if ((input_row >= 0l) && (input_row < (int32_t)input_rows) && (input_col >= 0l) && (input_col < (int32_t)input_cols))
                            {
                                sum += prev_layer->outputs[f]->values[input_row * input_cols + input_col] * layer->filters[d][f]->values[m * filter_cols + n];
                            }
                        }
                    }
                }

                output_value = sum + layer->filters_biases->values[d];

                // Apply activation function
                output_value = layer->activation_func(output_value);

                layer->outputs[d]->values[i * output_cols + j] = output_value;
            }
        }

        mbednn_normalize_feature_map(layer->outputs[d]->values, output_rows * output_cols, &gamma, &beta);
    }
}

// Perform max pooling on a 2D input layer and store the result in the 2D output layer (2x2 window is only supported)
static void mbednn_maxpool_2d_layer(layer_t *prev_layer, layer_t *layer)
{
    uint32_t input_rows = prev_layer->rows;
    uint32_t input_cols = prev_layer->cols;
    uint32_t output_rows = layer->rows;
    uint32_t output_cols = layer->cols;
    uint32_t input_i;
    uint32_t input_j;
    real max_value;
    real value;

    for (uint32_t d = 0ul; d < layer->depth; d++)
    {
        for (uint32_t i = 0ul; i < output_rows; i++)
        {
            for (uint32_t j = 0ul; j < output_cols; j++)
            {
                max_value = (real)(-INFINITY);
                for (uint32_t m = 0ul; m < 2ul; m++)
                {
                    for (uint32_t n = 0ul; n < 2ul; n++)
                    {
                        input_i = i * 2ul + m;
                        input_j = j * 2ul + n;

                        value = prev_layer->outputs[d]->values[input_i * input_cols + input_j];
                        if (value > max_value)
                        {
                            max_value = value;
                        }
                    }
                }

                layer->outputs[d]->values[i * output_cols + j] = max_value;
            }
        }
    }
}

// Perform average pooling on a 2D input layer and store the result in the 2D output layer
static void mbednn_avrpool_2d_layer(layer_t *prev_layer, layer_t *layer)
{
    uint32_t input_rows = prev_layer->rows;
    uint32_t input_cols = prev_layer->cols;
    uint32_t output_rows = layer->rows;
    uint32_t output_cols = layer->cols;
    uint32_t input_i;
    uint32_t input_j;
    real avr_value;

    for (uint32_t d = 0ul; d < layer->depth; d++)
    {
        for (uint32_t i = 0ul; i < output_rows; i++)
        {
            for (uint32_t j = 0ul; j < output_cols; j++)
            {
                avr_value = REAL_C(0.0);
                for (uint32_t m = 0ul; m < 2ul; m++)
                {
                    for (uint32_t n = 0ul; n < 2ul; n++)
                    {
                        input_i = i * 2ul + m;
                        input_j = j * 2ul + n;
                        avr_value += prev_layer->outputs[d]->values[input_i * input_cols + input_j];
                    }
                }

                layer->outputs[d]->values[i * output_cols + j] = avr_value / REAL_C(4.0);
            }
        }
    }
}

// Perform a forward pass through the network
static void mbednn_eval_network(mbednn_t *mbednn, bool training)
{
    real value;
    uint32_t dropout_count;
    layer_t *act_layer;
    layer_t *next_layer;

    if (mbednn != NULL)
    {
        // for all layers except the output layer
        for (uint32_t layer = 0ul; layer < (mbednn->layer_count - 1ul); layer++)
        {
            act_layer  = &mbednn->layers[layer];
            next_layer = &mbednn->layers[layer + 1ul];

            switch (next_layer->type)
            {
            case MBEDNN_LAYER_CONV_2D:
                mbednn_conv_2d_layer(act_layer, next_layer);
                break;

            case MBEDNN_LAYER_MAXPOOL_2D:
                mbednn_maxpool_2d_layer(act_layer, next_layer);
                break;

            case MBEDNN_LAYER_AVRPOOL_2D:
                mbednn_avrpool_2d_layer(act_layer, next_layer);
                break;

            case MBEDNN_LAYER_FLATTEN_2D:
                // flatten layer, just copy the inputs to the outputs
                for (uint32_t d = 0ul; d < act_layer->depth; d++)
                {
                    memcpy(&next_layer->outputs[0]->values[d * act_layer->node_count], act_layer->outputs[d]->values, act_layer->node_count * sizeof(real));
                }
                break;

            case MBEDNN_LAYER_DROPOUT:
                if (training == true)
                {
                    // training mode, determine the number of nodes to randomly drop
                    dropout_count = (uint32_t)(next_layer->dropout_rate * act_layer->node_count);

                    for (uint32_t node = 0ul; node < act_layer->node_count; node++)
                    {
                        uint32_t i = next_layer->input_indices[node];

                        if (node < dropout_count)
                        {
                            // drop these nodes
                            next_layer->dropout_mask->values[i] = REAL_C(0.0);
                            next_layer->outputs[0]->values[i] = REAL_C(0.0);
                        }
                        else
                        {
                            // keep these nodes and apply scaling
                            next_layer->dropout_mask->values[i] = REAL_C(1.0) / (REAL_C(1.0) - next_layer->dropout_rate);
                            next_layer->outputs[0]->values[i] = act_layer->outputs[0]->values[i] * next_layer->dropout_mask->values[i];
                        }
                    }
                }
                else
                {
                    // predict mode, just copy the inputs to the outputs
                    memcpy(next_layer->outputs[0]->values, act_layer->outputs[0]->values, act_layer->node_count * sizeof(real));
                }
                break;

            case MBEDNN_LAYER_DENSE:
            case MBEDNN_LAYER_OUTPUT:
                matrix_matrix_vector_product(REAL_C(1.0), next_layer->weights, REAL_C(0.0), act_layer->outputs[0], next_layer->outputs[0]);
                matrix_ax_add_by(REAL_C(1.0), next_layer->biases, REAL_C(1.0), next_layer->outputs[0]);

                // apply the activation and derivation function to the output
                for (uint32_t node = 0ul; node < next_layer->node_count; node++)
                {
                    value = next_layer->outputs[0]->values[node];
                    next_layer->outputs[0]->values[node] = next_layer->activation_func(value);
                    next_layer->derivatives[0]->values[node] = next_layer->derivation_func(value);
                }
                break;

            default:
                break;
            }
        }

        // apply the softmax on the output if requested
        if (mbednn->layers[mbednn->layer_count - 1ul].activation_type == MBEDNN_ACTIVATION_SOFTMAX)
        {
            mbednn_softmax(mbednn);
        }
    }
}

#ifdef MBEDNN_USE_TRAINING
// Back propagate the output layer to the previous layer
// This function calculates the gradient of the loss with respect to the output layer's weights and biases,
// and propagates this gradient back to the previous layer. It updates the biases and gradients of the previous layer
// and computes the gradient of the loss with respect to the previous layer's outputs.
static void mbednn_back_propagate_output_layer(mbednn_t *mbednn, layer_t *layer, layer_t *prev_layer, matrix_t *outputs)
{
    // Step 1: compute the error (loss derivative) w.r.t the layer's output
    // dL/dy = y_pred - y_true
    for (uint32_t j = 0ul; j < layer->node_count; j++)
    {
        layer->dl_dz[0ul]->values[j] = layer->outputs[0ul]->values[j] - outputs->values[j];
    }

    // Step 2: compute gradients for weights
    // weights_gradients = h^T * dL/dy
    matrix_vector_outer_product(REAL_C(1.0), layer->dl_dz[0ul], prev_layer->outputs[0ul], layer->weights_gradients);

    // Step 3: compute gradients for biases
    // biases_gradients = dL/dy
    matrix_ax_add_by(REAL_C(1.0), layer->dl_dz[0ul], REAL_C(1.0), layer->biases_gradients);

    // Step 4: backpropagate the gradient to the previous layer
    // dL/dz = dL/dy * weights^T
    matrix_matrix_vector_transposed_product(REAL_C(1.0), layer->weights, REAL_C(0.0), layer->dl_dz[0ul], prev_layer->dl_dz[0ul]);
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Back propagate dense layers to the previous layer and update the gradients and biases
static void mbednn_back_propagate_dense_layer(mbednn_t *mbednn, layer_t *layer, layer_t *prev_layer)
{
    // Step 1: compute the gradient of the loss w.r.t the layer's output
    // dl_dz = dl_dz * derivatives
    matrix_hadamard_product(layer->dl_dz[0ul], layer->derivatives[0ul]);

    // Step 2: compute gradients for weights
    // weights_gradients = h^T * dL/dz
    matrix_vector_outer_product(REAL_C(1.0), layer->dl_dz[0ul], prev_layer->outputs[0ul], layer->weights_gradients);

    // Step 3: compute gradients for biases
    // biases_gradients = dL/dz
    matrix_ax_add_by(REAL_C(1.0), layer->dl_dz[0ul], REAL_C(1.0), layer->biases_gradients);

    // Step 4: backpropagate the gradient to the previous layer
    // dL/dz = dL/dz * weights^T
    matrix_matrix_vector_transposed_product(REAL_C(1.0), layer->weights, REAL_C(0.0), layer->dl_dz[0ul], prev_layer->dl_dz[0ul]);
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Back propagate flatten layer to the previous layer for each channel, passing gradients through without modification
static void mbednn_back_propagate_flatten_layer(mbednn_t *mbednn, layer_t *layer, layer_t *prev_layer)
{
    for (uint32_t d = 0ul; d < prev_layer->depth; d++)
    {
        memcpy(prev_layer->dl_dz[d]->values, &layer->dl_dz[0]->values[d * prev_layer->node_count], prev_layer->node_count * sizeof(real));
    }
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Back propagate dropout layer to the previous layer for each channel, passing gradients through without modification
static void mbednn_back_propagate_dropout_layer(mbednn_t* mbednn, layer_t* layer, layer_t* prev_layer)
{
    for (uint32_t node = 0ul; node < prev_layer->node_count; node++)
    {
        prev_layer->dl_dz[0]->values[node] = layer->dl_dz[0]->values[node] * layer->dropout_mask->values[node];
    }
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Handle backpropagation for convolution layers specifically by propagating gradients to the previous layer
static void mbednn_back_propagate_conv_layer(mbednn_t *mbednn, layer_t *layer, layer_t *prev_layer)
{
    uint32_t input_rows = prev_layer->rows;
    uint32_t input_cols = prev_layer->cols;
    uint32_t output_rows = layer->rows;
    uint32_t output_cols = layer->cols;
    uint32_t filter_rows = layer->filters_rows;
    uint32_t filter_cols = layer->filters_cols;

    // Compute gradients for filters and biases
    for (uint32_t d = 0ul; d < layer->depth; d++)
    {
        for (uint32_t i = 0ul; i < output_rows; i++)
        {
            for (uint32_t j = 0ul; j < output_cols; j++)
            {
                real dl_dz = layer->dl_dz[d]->values[i * output_cols + j];

                // Apply the derivative of the activation function
                dl_dz *= layer->derivation_func(layer->outputs[d]->values[i * output_cols + j]);

                for (uint32_t f = 0ul; f < layer->filters_count; f++)
                {
                    for (uint32_t m = 0ul; m < filter_rows; m++)
                    {
                        for (uint32_t n = 0ul; n < filter_cols; n++)
                        {
                            int32_t input_row = (int32_t)(i * layer->filters_stride) + (int32_t)m - (int32_t)layer->filters_top_padding;
                            int32_t input_col = (int32_t)(j * layer->filters_stride) + (int32_t)n - (int32_t)layer->filters_left_padding;

                            // Check if the filter overlaps the border
                            if ((input_row >= 0l) && (input_row < (int32_t)input_rows) && (input_col >= 0l) && (input_col < (int32_t)input_cols))
                            {
                                layer->filters_gradients[d][f]->values[m * filter_cols + n] += dl_dz * prev_layer->outputs[f]->values[input_row * input_cols + input_col];
                            }

                        }
                    }
                }
                layer->filters_biases_gradients->values[d] += dl_dz;
            }
        }
    }

    // Propagate gradients to the previous layer
    for (uint32_t f = 0ul; f < layer->filters_count; f++)
    {
        for (uint32_t i = 0ul; i < input_rows; i++)
        {
            for (uint32_t j = 0ul; j < input_cols; j++)
            {
                real sum = REAL_C(0.0);
                for (uint32_t d = 0ul; d < layer->depth; d++)
                {
                    for (uint32_t m = 0ul; m < filter_rows; m++)
                    {
                        for (uint32_t n = 0ul; n < filter_cols; n++)
                        {
                            int32_t input_row = (int32_t)i + (int32_t)m - (int32_t)layer->filters_top_padding;
                            int32_t input_col = (int32_t)j + (int32_t)n - (int32_t)layer->filters_left_padding;
                            int32_t output_row = input_row / (int32_t)layer->filters_stride;
                            int32_t output_col = input_col / (int32_t)layer->filters_stride;

                            // Check if the filter overlaps the border and if the indices are valid
                            if ((output_row >= 0l) && (output_row < (int32_t)output_rows) && (output_col >= 0l) && (output_col < (int32_t)output_cols) &&
                                ((input_row % (int32_t)layer->filters_stride) == 0l) && ((input_col % (int32_t)layer->filters_stride) == 0l))
                            {
                                sum += layer->dl_dz[d]->values[output_row * output_cols + output_col] * layer->filters[d][f]->values[m * filter_cols + n];
                            }
                        }
                    }
                }
                prev_layer->dl_dz[f]->values[i * input_cols + j] += sum;
            }
        }
    }
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Handle backpropagation for maxpool layers specifically by propagating gradients to the previous layer
static void mbednn_back_propagate_maxpool_layer(mbednn_t *mbednn, layer_t *layer, layer_t *prev_layer)
{
    uint32_t input_rows = prev_layer->rows;
    uint32_t input_cols = prev_layer->cols;
    uint32_t output_rows = layer->rows;
    uint32_t output_cols = layer->cols;
    uint32_t input_index;
    real max_value;
    bool max_found;

    for (uint32_t d = 0ul; d < layer->depth; d++)
    {
        matrix_fill_value(prev_layer->dl_dz[d], REAL_C(0.0));
    }

    for (uint32_t d = 0ul; d < layer->depth; d++)
    {
        for (uint32_t i = 0ul; i < output_rows; i++)
        {
            for (uint32_t j = 0ul; j < output_cols; j++)
            {
                max_value = layer->outputs[d]->values[i * output_cols + j];
                max_found = false;

                for (uint32_t m = 0ul; m < 2ul; m++)
                {
                    for (uint32_t n = 0ul; n < 2ul; n++)
                    {
                        input_index = (2ul * i + m) * input_cols + (2ul * j + n);
                        if ((prev_layer->outputs[d]->values[input_index] == max_value) && (max_found == false))
                        {
                            prev_layer->dl_dz[d]->values[input_index] = layer->dl_dz[d]->values[i * output_cols + j];
                            max_found = true;
                        }
                    }
                }
            }
        }
    }
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Handle backpropagation for avrpool layers specifically by propagating gradients to the previous layer
static void mbednn_back_propagate_avrpool_layer(mbednn_t *mbednn, layer_t *layer, layer_t *prev_layer)
{
    uint32_t input_rows = prev_layer->rows;
    uint32_t input_cols = prev_layer->cols;
    uint32_t output_rows = layer->rows;
    uint32_t output_cols = layer->cols;
    uint32_t input_index;

    for (uint32_t d = 0ul; d < layer->depth; d++)
    {
        for (uint32_t i = 0ul; i < output_rows; i++)
        {
            for (uint32_t j = 0ul; j < output_cols; j++)
            {
                for (uint32_t m = 0ul; m < 2ul; m++)
                {
                    for (uint32_t n = 0ul; n < 2ul; n++)
                    {
                        input_index = (2ul * i + m) * input_cols + (2ul * j + n);
                        prev_layer->dl_dz[d]->values[input_index] = layer->dl_dz[d]->values[i * output_cols + j] / REAL_C(4.0);
                    }
                }
            }
        }
    }
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Compute the gradients via back propagation
static void mbednn_back_propagate(mbednn_t *mbednn, matrix_t *outputs)
{
    uint32_t output_layer = mbednn->layer_count - 1ul;

    // output layer back propagation 
    mbednn_back_propagate_output_layer(mbednn, &mbednn->layers[output_layer], &mbednn->layers[output_layer - 1ul], outputs);

    // dense layer back propagation excluding the input layer
    for (uint32_t layer = output_layer - 1ul; layer > 0ul; layer--)
    {
        switch(mbednn->layers[layer].type)
        {
        case MBEDNN_LAYER_CONV_2D:
            mbednn_back_propagate_conv_layer(mbednn, &mbednn->layers[layer], &mbednn->layers[layer - 1ul]);
            break;

        case MBEDNN_LAYER_MAXPOOL_2D:
            mbednn_back_propagate_maxpool_layer(mbednn, &mbednn->layers[layer], &mbednn->layers[layer - 1ul]);
            break;

        case MBEDNN_LAYER_AVRPOOL_2D:
            mbednn_back_propagate_avrpool_layer(mbednn, &mbednn->layers[layer], &mbednn->layers[layer - 1ul]);
            break;

        case MBEDNN_LAYER_FLATTEN_2D:
            mbednn_back_propagate_flatten_layer(mbednn, &mbednn->layers[layer], &mbednn->layers[layer - 1ul]);
            break;

        case MBEDNN_LAYER_DROPOUT:
            mbednn_back_propagate_dropout_layer(mbednn, &mbednn->layers[layer], &mbednn->layers[layer - 1ul]);
            break;

        case MBEDNN_LAYER_DENSE:
            mbednn_back_propagate_dense_layer(mbednn, &mbednn->layers[layer], &mbednn->layers[layer - 1ul]);
            break;

        default:
            break;
        }
    }
}
#endif

// Allocate a new layer
static int32_t mbednn_allocate_layer(mbednn_t* mbednn)
{
    int32_t ret = MBEDNN_NOK;
    layer_t* layer;

    mbednn->layer_count++;
    layer = alloc_aligned_malloc(mbednn->layer_count * (sizeof(layer_t)), 8);
    if (layer != NULL)
    {
        if (mbednn->layers != NULL)
        {
            memcpy(layer, mbednn->layers, (mbednn->layer_count - 1ul) * (sizeof(layer_t)));
            alloc_aligned_free(mbednn->layers);
        }
        mbednn->layers = layer;

        ret = MBEDNN_OK;
    }

    return ret;
}

#ifdef MBEDNN_USE_TRAINING
// Train the network with a single input set and output set
static real mbednn_train_pass_network(mbednn_t *mbednn, matrix_t **inputs, matrix_t *outputs)
{
    real loss = REAL_C(0.0);

    if ((mbednn != NULL) && (inputs != NULL) && (outputs != NULL))
    {
        if (((mbednn->layers[0ul].type == MBEDNN_LAYER_INPUT) || (mbednn->layers[0ul].type == MBEDNN_LAYER_INPUT_2D)) && 
            (mbednn->layers[mbednn->layer_count - 1ul].type == MBEDNN_LAYER_OUTPUT))
        {
            // set the input values on the network
            for (uint32_t d = 0ul; d < mbednn->layers[0ul].depth; d++)
            {
                for (uint32_t node = 0ul; node < mbednn->layers[0ul].node_count; node++)
                {
                    mbednn->layers[0ul].outputs[d]->values[node] = inputs[d]->values[node];
                }
            }

            // forward evaluation of the network
            mbednn_eval_network(mbednn, true);

            // compute the loss
            loss = mbednn->loss_func(mbednn, outputs);

            // back propagate error through the network to compute the gradients
            mbednn_back_propagate(mbednn, outputs);
        }
    }

    return loss;
}
#endif

#ifdef MBEDNN_USE_TRAINING
// This function reduces the learning rate by a decay factor
static void mbednn_optimize_decay(mbednn_t *mbednn, real loss)
{
    mbednn->learning_rate *= MBEDNN_DEFAULT_LEARNING_DECAY;
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Adaptive learning rate
static void mbednn_optimize_adapt(mbednn_t *mbednn, real loss)
{
    real lastMSE = REAL_C(0.0);

    // adapt the learning rate
    mbednn_optimize_decay(mbednn, loss);

    // average the last n learning rates
    for (uint32_t i = 0ul; i < MBEDNN_DEFAULT_MSE_AVG; i++)
    {
        lastMSE += mbednn->last_mse[i];
    }
    lastMSE /= (real)MBEDNN_DEFAULT_MSE_AVG;

    if (lastMSE > REAL_C(0.0))
    {
        if (loss < lastMSE)
        {
            mbednn->learning_rate += MBEDNN_DEFAULT_LEARN_ADD;

            // don't let learning rate go above 1
            if (mbednn->learning_rate > REAL_C(1.0))
            {
                mbednn->learning_rate = REAL_C(1.0);
            }
        }
        else
        {
            mbednn->learning_rate -= MBEDNN_DEFAULT_LEARN_SUB * mbednn->learning_rate;

            // don't let learning rate go below 0
            if (mbednn->learning_rate < REAL_C(0.0))
            {
                mbednn->learning_rate = REAL_C(0.0);
            }
        }
    }

    mbednn->last_mse[mbednn->mse_counter] = loss;
    mbednn->mse_counter = (mbednn->mse_counter + 1ul) % MBEDNN_DEFAULT_MSE_AVG;
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Stochastic Gradient Descent
static void mbednn_optimize_sgd(mbednn_t *mbednn)
{
    // No update for input layers necessary
    for (uint32_t layer = 1ul; layer < mbednn->layer_count; layer++)
    {
        switch (mbednn->layers[layer].type)
        {
        case MBEDNN_LAYER_CONV_2D:
            // Update convolutional filters and biases
            for (uint32_t d = 0ul; d < mbednn->layers[layer].depth; d++)
            {
                for (uint32_t f = 0ul; f < mbednn->layers[layer].filters_count; f++)
                {
                    matrix_ax_add_by(-mbednn->learning_rate, mbednn->layers[layer].filters_gradients[d][f], REAL_C(1.0), mbednn->layers[layer].filters[d][f]);
                }
            }
            matrix_ax_add_by(-mbednn->learning_rate, mbednn->layers[layer].filters_biases_gradients, REAL_C(1.0), mbednn->layers[layer].filters_biases);
            break;
        case MBEDNN_LAYER_DENSE:
        case MBEDNN_LAYER_OUTPUT:
            // Update weights and biases
            matrix_ax_add_by(-mbednn->learning_rate, mbednn->layers[layer].weights_gradients, REAL_C(1.0), mbednn->layers[layer].weights);
            matrix_ax_add_by(-mbednn->learning_rate, mbednn->layers[layer].biases_gradients, REAL_C(1.0), mbednn->layers[layer].biases);
            break;
        default:
            break;
        }
    }
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Stochastic Gradient Descent with Momentum
static void mbednn_optimize_momentum(mbednn_t *mbednn)
{
    real beta = REAL_C(0.9);
    real one_minus_beta = REAL_C(0.1);

    // No update for input layers necessary
    for (uint32_t layer = 1ul; layer < mbednn->layer_count; layer++)
    {
        switch (mbednn->layers[layer].type)
        {
        case MBEDNN_LAYER_CONV_2D:
            // Update convolutional filters and biases
            for (uint32_t d = 0ul; d < mbednn->layers[layer].depth; d++)
            {
                for (uint32_t f = 0ul; f < mbednn->layers[layer].filters_count; f++)
                {
                    matrix_ax_add_by(-mbednn->learning_rate, mbednn->layers[layer].filters_gradients[d][f], REAL_C(1.0), mbednn->layers[layer].filters[d][f]);
                }
            }
            matrix_ax_add_by(-mbednn->learning_rate, mbednn->layers[layer].filters_biases_gradients, REAL_C(1.0), mbednn->layers[layer].filters_biases);
            break;
        case MBEDNN_LAYER_DENSE:
        case MBEDNN_LAYER_OUTPUT:
            // Update momentums for weights: momentum = beta * momentum + one_minus_beta * gradients
            matrix_ax_add_by(one_minus_beta, mbednn->layers[layer].weights_gradients, beta, mbednn->layers[layer].momentums);

            // Update weights: weights = weights - n * momentum
            matrix_ax_add_by(-mbednn->learning_rate, mbednn->layers[layer].momentums, REAL_C(1.0), mbednn->layers[layer].weights);

            // Update momentums for biases: bias_momentum = beta * bias_momentum + one_minus_beta * bias_gradients
            matrix_ax_add_by(one_minus_beta, mbednn->layers[layer].biases_gradients, beta, mbednn->layers[layer].bias_momentums);

            // Update biases: biases = biases - n * bias_momentum
            matrix_ax_add_by(-mbednn->learning_rate, mbednn->layers[layer].bias_momentums, REAL_C(1.0), mbednn->layers[layer].biases);
            break;
        default:
            break;
        }
    }
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Adaptive Gradient Algorithm (Adagrad) optimizer implementation
static void mbednn_optimize_adagrad(mbednn_t *mbednn)
{
    real gradient;
    real value;

    // for each layer except the input layer
    for (uint32_t layer = 1ul; layer < mbednn->layer_count; layer++)
    {
        switch (mbednn->layers[layer].type)
        {
        case MBEDNN_LAYER_CONV_2D:
            // Update convolutional filters and biases
            for (uint32_t d = 0ul; d < mbednn->layers[layer].depth; d++)
            {
                for (uint32_t f = 0ul; f < mbednn->layers[layer].filters_count; f++)
                {
                    matrix_ax_add_by(-mbednn->learning_rate, mbednn->layers[layer].filters_gradients[d][f], REAL_C(1.0), mbednn->layers[layer].filters[d][f]);
                }
            }
            matrix_ax_add_by(-mbednn->learning_rate, mbednn->layers[layer].filters_biases_gradients, REAL_C(1.0), mbednn->layers[layer].filters_biases);
            break;
        case MBEDNN_LAYER_DENSE:
        case MBEDNN_LAYER_OUTPUT:
            // update weights
            for (uint32_t i = 0ul; i < (mbednn->layers[layer].weights->rows * mbednn->layers[layer].weights->cols); i++)
            {
                gradient = mbednn->layers[layer].weights_gradients->values[i];

                // accumulate squared gradients
                mbednn->layers[layer].velocities->values[i] += gradient * gradient;

                // update weights
                value = REAL_FMAX(REAL_SQRT(mbednn->layers[layer].velocities->values[i]), REAL_DIV_EPSILON);
                mbednn->layers[layer].weights->values[i] -= mbednn->learning_rate * gradient / value;
            }

            // update biases
            for (uint32_t i = 0ul; i < mbednn->layers[layer].biases->cols; i++)
            {
                gradient = mbednn->layers[layer].biases_gradients->values[i];

                // accumulate squared gradients
                mbednn->layers[layer].bias_velocities->values[i] += gradient * gradient;

                // update biases
                value = REAL_FMAX(REAL_SQRT(mbednn->layers[layer].bias_velocities->values[i]), REAL_DIV_EPSILON);
                mbednn->layers[layer].biases->values[i] -= mbednn->learning_rate * gradient / value;
            }
            break;
        default:
            break;
        }
    }
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Root Mean Square Propagation (RMSProp)
// RMSProp is an adaptive learning rate optimization algorithm that maintains a moving average of the squared gradients
// and divides the gradient by the root of this average. This helps to normalize the gradient and prevent large updates.
static void mbednn_optimize_rmsprop(mbednn_t *mbednn)
{
    real beta = REAL_C(0.9);
    real one_minus_beta = REAL_C(0.1);
    real gradient;
    real value;

    // for each layer except the input layer
    for (uint32_t layer = 1ul; layer < mbednn->layer_count; layer++)
    {
        switch (mbednn->layers[layer].type)
        {
        case MBEDNN_LAYER_CONV_2D:
            // Update convolutional filters and biases
            for (uint32_t d = 0ul; d < mbednn->layers[layer].depth; d++)
            {
                for (uint32_t f = 0ul; f < mbednn->layers[layer].filters_count; f++)
                {
                    matrix_ax_add_by(-mbednn->learning_rate, mbednn->layers[layer].filters_gradients[d][f], REAL_C(1.0), mbednn->layers[layer].filters[d][f]);
                }
            }
            matrix_ax_add_by(-mbednn->learning_rate, mbednn->layers[layer].filters_biases_gradients, REAL_C(1.0), mbednn->layers[layer].filters_biases);
            break;
        case MBEDNN_LAYER_DENSE:
        case MBEDNN_LAYER_OUTPUT:
            // update weights
            for (uint32_t i = 0ul; i < (mbednn->layers[layer].weights->rows * mbednn->layers[layer].weights->cols); i++)
            {
                gradient = mbednn->layers[layer].weights_gradients->values[i];

                // update the velocities (squared gradient moving average)
                mbednn->layers[layer].velocities->values[i] = beta * mbednn->layers[layer].velocities->values[i] + one_minus_beta * gradient * gradient;

                // update weights
                value = REAL_FMAX(REAL_SQRT(mbednn->layers[layer].velocities->values[i]), REAL_DIV_EPSILON);
                mbednn->layers[layer].weights->values[i] -= mbednn->learning_rate * gradient / value;
            }

            // update biases
            for (uint32_t i = 0ul; i < mbednn->layers[layer].biases->cols; i++)
            {
                gradient = mbednn->layers[layer].biases_gradients->values[i];

                // update the squared gradient moving average
                mbednn->layers[layer].bias_velocities->values[i] = mbednn->layers[layer].bias_velocities->values[i] + one_minus_beta * gradient * gradient;

                // update biases
                value = REAL_FMAX(REAL_SQRT(mbednn->layers[layer].bias_velocities->values[i]), REAL_DIV_EPSILON);
                mbednn->layers[layer].biases->values[i] -= mbednn->learning_rate * gradient / value;
            }
            break;
        default:
            break;
        }
    }
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Adaptive Moment Estimation (Adam) optimizer
// Adam is an optimization algorithm that computes adaptive learning rates for each parameter.
// It uses estimates of first and second moments of the gradients to adapt the learning rate for each weight of the neural network.
static void mbednn_optimize_adam(mbednn_t *mbednn)
{
    real beta1 = REAL_C(0.9);
    real one_minus_beta1 = REAL_C(0.1);
    real beta2 = REAL_C(0.999);
    real one_minus_beta2 = REAL_C(0.001);
    real gradient;
    real value;
    real value1;
    real value2;
    real m_hat;
    real v_hat;

    mbednn->train_iteration++;
    value1 = REAL_FMAX(REAL_C(1.0) - REAL_POW(beta1, (real)(mbednn->train_iteration)), REAL_DIV_EPSILON);
    value2 = REAL_FMAX(REAL_C(1.0) - REAL_POW(beta2, (real)(mbednn->train_iteration)), REAL_DIV_EPSILON);

    // for each layer except the input layer
    for (uint32_t layer = 1ul; layer < mbednn->layer_count; layer++)
    {
        switch (mbednn->layers[layer].type)
        {
        case MBEDNN_LAYER_CONV_2D:
            // Update convolutional filters and biases
            for (uint32_t d = 0ul; d < mbednn->layers[layer].depth; d++)
            {
                for (uint32_t f = 0ul; f < mbednn->layers[layer].filters_count; f++)
                {
                    matrix_ax_add_by(-mbednn->learning_rate, mbednn->layers[layer].filters_gradients[d][f], REAL_C(1.0), mbednn->layers[layer].filters[d][f]);
                }
            }
            matrix_ax_add_by(-mbednn->learning_rate, mbednn->layers[layer].filters_biases_gradients, REAL_C(1.0), mbednn->layers[layer].filters_biases);
            break;
        case MBEDNN_LAYER_DENSE:
        case MBEDNN_LAYER_OUTPUT:
            // update weights
            for (uint32_t i = 0ul; i < (mbednn->layers[layer].weights->rows * mbednn->layers[layer].weights->cols); i++)
            {
                gradient = mbednn->layers[layer].weights_gradients->values[i];

                // compute biased first moment estimate (momentum)
                mbednn->layers[layer].momentums->values[i] = beta1 * mbednn->layers[layer].momentums->values[i] + one_minus_beta1 * gradient;

                // compute biased second moment estimate (RMSProp-like variance)
                mbednn->layers[layer].velocities->values[i] = beta2 * mbednn->layers[layer].velocities->values[i] + one_minus_beta2 * gradient * gradient;

                // correct bias for first and second moment
                m_hat = mbednn->layers[layer].momentums->values[i] / value1;
                v_hat = mbednn->layers[layer].velocities->values[i] / value2;

                // update weights
                value = REAL_FMAX(REAL_SQRT(v_hat), REAL_DIV_EPSILON);
                mbednn->layers[layer].weights->values[i] -= mbednn->learning_rate * m_hat / value;
            }

            // update biases
            for (uint32_t i = 0ul; i < mbednn->layers[layer].biases->cols; i++)
            {
                gradient = mbednn->layers[layer].biases_gradients->values[i];

                // compute biased first moment estimate for biases
                mbednn->layers[layer].bias_momentums->values[i] = beta1 * mbednn->layers[layer].bias_momentums->values[i] + one_minus_beta1 * gradient;

                // compute biased second moment estimate for biases
                mbednn->layers[layer].bias_velocities->values[i] = beta2 * mbednn->layers[layer].bias_velocities->values[i] + one_minus_beta2 * gradient * gradient;

                // correct bias for first and second moment
                m_hat = mbednn->layers[layer].bias_momentums->values[i] / value1;
                v_hat = mbednn->layers[layer].bias_velocities->values[i] / value2;

                // update biases
                value = REAL_FMAX(REAL_SQRT(v_hat), REAL_DIV_EPSILON);
                mbednn->layers[layer].biases->values[i] -= mbednn->learning_rate * m_hat / value;
            }
            break;
        default:
            break;
        }
    }
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Zero the gradients for all layers
static void mbednn_zero_gradients(mbednn_t *mbednn)
{
    for (uint32_t layer = 0ul; layer < mbednn->layer_count; layer++)
    {
        switch (mbednn->layers[layer].type)
        {
        case MBEDNN_LAYER_CONV_2D:
            for (uint32_t d = 0ul; d < mbednn->layers[layer].depth; d++)
            {
                for (uint32_t f = 0ul; f < mbednn->layers[layer].filters_count; f++)
                {
                    matrix_fill_value(mbednn->layers[layer].filters_gradients[d][f], REAL_C(0.0));
                }
            }
            matrix_fill_value(mbednn->layers[layer].filters_biases_gradients, REAL_C(0.0));
            break;

        case MBEDNN_LAYER_DENSE:
        case MBEDNN_LAYER_OUTPUT:
            matrix_fill_value(mbednn->layers[layer].weights_gradients, REAL_C(0.0));
            matrix_fill_value(mbednn->layers[layer].biases_gradients, REAL_C(0.0));
            matrix_fill_value(mbednn->layers[layer].momentums, REAL_C(0.0));
            matrix_fill_value(mbednn->layers[layer].bias_momentums, REAL_C(0.0));
            matrix_fill_value(mbednn->layers[layer].velocities, REAL_C(0.0));
            matrix_fill_value(mbednn->layers[layer].bias_velocities, REAL_C(0.0));
            break;

        default:
            break;
        }
    }
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Shuffle node indices for dropout layers
static void mbednn_shuffle_dropout_indices(mbednn_t* mbednn)
{
    for (uint32_t layer = 0ul; layer < mbednn->layer_count; layer++)
    {
        if (mbednn->layers[layer].type == MBEDNN_LAYER_DROPOUT)
        {
            mbednn_shuffle_indices(mbednn->layers[layer].input_indices, mbednn->layers[layer].node_count);
        }
    }
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Zero the gradients for all layers
static void mbednn_zero_dL_dz(mbednn_t* mbednn)
{
    for (uint32_t layer = 0ul; layer < mbednn->layer_count; layer++)
    {
        switch (mbednn->layers[layer].type)
        {
        case MBEDNN_LAYER_INPUT_2D:
        case MBEDNN_LAYER_CONV_2D:
        case MBEDNN_LAYER_MAXPOOL_2D:
        case MBEDNN_LAYER_AVRPOOL_2D:
            for (uint32_t d = 0ul; d < mbednn->layers[layer].depth; d++)
            {
                matrix_fill_value(mbednn->layers[layer].dl_dz[d], REAL_C(0.0));
            }
            break;

        case MBEDNN_LAYER_FLATTEN_2D:
        case MBEDNN_LAYER_INPUT:
        case MBEDNN_LAYER_DENSE:
        case MBEDNN_LAYER_DROPOUT:
        case MBEDNN_LAYER_OUTPUT:
            matrix_fill_value(mbednn->layers[layer].dl_dz[0], REAL_C(0.0));
            break;

        default:
            break;
        }
    }
}
#endif

#ifdef MBEDNN_USE_TRAINING
// Print the progress
static void mbednn_print_progress(uint32_t max_cnt, uint32_t cnt, uint32_t max_row, uint32_t row, real loss_batch)
{
    static char progress[4ul] = { '-', '\\', '|', '/' };

    MBEDNN_PRINTF("\r(%c) %7.3f%% Epoch=%05d LossPerBatch=%6.4f", progress[row % 4ul], 100.0f * ((real)(cnt * max_row + row)) / ((real)(max_cnt * max_row)), cnt, loss_batch);
}
#endif

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_create
**
**     brief    Create a new neural network. Network variables 
**              are initialized to default values. Random number generator is 
**              seeded.
**
**     params   none
**     return   pointer to the created neural network
*/
/*---------------------------------------------------------------------------*/
mbednn_t* mbednn_create(void)
{
    mbednn_t *mbednn = alloc_aligned_malloc(sizeof(mbednn_t), 8);

    if (mbednn != NULL)
    {
        // seed the random number generator
        random_init();

        mbednn->layers = NULL;
        mbednn->layer_count = 0ul;

        mbednn->mse_counter = 0ul;
        for (uint32_t i = 0ul; i < MBEDNN_DEFAULT_MSE_AVG; i++)
        {
            mbednn->last_mse[i] = REAL_C(0.0);
        }

        mbednn->epsilon = MBEDNN_DEFAULT_EPSILON;
        mbednn->learning_rate = MBEDNN_DEFAULT_LEARNING_RATE;
        mbednn->loss_type = MBEDNN_LOSS_DEFAULT;
        mbednn->loss_func = NULL;
        mbednn->optimizer_type = MBEDNN_OPTIMIZER_DEFAULT;
        mbednn->optimization_func = NULL;
        mbednn->train_iteration = 0ul;
        mbednn->epoch_limit = MBEDNN_DEFAULT_EPOCHS;
        mbednn->batch_size = MBEDNN_DEFAULT_BATCH_SIZE;
    }

    return mbednn;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_add_layer_input
**
**     brief    Add an input layer to a neural network. Layer variables are
**              initialized and related matrices are created.
**
**     params   mbednn: pointer to the neural network
**              node_count: number of neurons (nodes) in that layer
**     return   layer number if the layer is added successfully, otherwise MBEDNN_NOK
*/
/*---------------------------------------------------------------------------*/
uint32_t mbednn_add_layer_input(mbednn_t *mbednn, uint32_t node_count)
{
    uint32_t ret = (uint32_t)MBEDNN_NOK;
    uint32_t cur_layer;
    uint32_t allocated_bytes = (uint32_t)alloc_get_bytes_allocated();

    // parameter and plausibility checks, early returns when failed
    if (mbednn == NULL) return ret;
    if (mbednn->layer_count > 0ul) return ret;

    // allocate a new layer
    if (mbednn_allocate_layer(mbednn) == MBEDNN_OK)
    {
        // set parameters of the layer
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].type = MBEDNN_LAYER_INPUT;
        strcpy(mbednn->layers[cur_layer].type_name, "Input");
        mbednn->layers[cur_layer].random_type = MBEDNN_RANDOM_NONE;
        mbednn->layers[cur_layer].activation_type = MBEDNN_ACTIVATION_NULL;
        mbednn->layers[cur_layer].activation_func = mbednn_null_activation;
        mbednn->layers[cur_layer].derivation_func = mbednn_null_derivation;
        mbednn->layers[cur_layer].rows = 1ul;
        mbednn->layers[cur_layer].cols = node_count;
        mbednn->layers[cur_layer].node_count = node_count;
        mbednn->layers[cur_layer].depth = 1ul;
        mbednn->layers[cur_layer].filters_count = 0ul;

        // create the matrices of the layer
        mbednn->layers[cur_layer].outputs = (matrix_t**)alloc_aligned_malloc(sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].outputs[0] = matrix_create_zeros(1ul, node_count);
#ifdef MBEDNN_USE_TRAINING
        mbednn->layers[cur_layer].derivatives = (matrix_t**)alloc_aligned_malloc(sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].derivatives[0] = matrix_create_zeros(1ul, node_count);
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)alloc_aligned_malloc(sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].dl_dz[0] = matrix_create_zeros(1ul, node_count);
#else
        mbednn->layers[cur_layer].derivatives = (matrix_t**)NULL;
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)NULL;
#endif
        // matrices which are not used for current layer type
        mbednn->layers[cur_layer].weights = NULL;
        mbednn->layers[cur_layer].weights_gradients = NULL;
        mbednn->layers[cur_layer].biases = NULL;
        mbednn->layers[cur_layer].biases_gradients = NULL;
        mbednn->layers[cur_layer].velocities = NULL;
        mbednn->layers[cur_layer].bias_velocities = NULL;
        mbednn->layers[cur_layer].momentums = NULL;
        mbednn->layers[cur_layer].bias_momentums = NULL;

        mbednn->layers[cur_layer].allocated_bytes = (uint32_t)alloc_get_bytes_allocated() - allocated_bytes;
        mbednn->layers[cur_layer].trainable_parameters = 0ul;
        ret = cur_layer;
    }
    else
    {
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].allocated_bytes = 0ul;
        mbednn->layers[cur_layer].trainable_parameters = 0ul;
    }

    return ret;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_add_layer_dense
**
**     brief    Add a dense layer to a neural network. Layer variables are
**              initialized and related matrices are created.
**
**     params   mbednn: pointer to the neural network
**              node_count: number of neurons (nodes) in that layer
**              activation_type: e.g. ReLU, Sigmoid, Tanh, etc.
**              random_type: e.g. uniform, normal, glorot, he, etc.
**     return   layer number if the layer is added successfully, otherwise MBEDNN_NOK
*/
/*---------------------------------------------------------------------------*/
uint32_t mbednn_add_layer_dense(mbednn_t *mbednn, uint32_t node_count, mbednn_activation_type activation_type, mbednn_random_type random_type)
{
    uint32_t ret = (uint32_t)MBEDNN_NOK;
    uint32_t cur_layer;
    uint32_t allocated_bytes = (uint32_t)alloc_get_bytes_allocated();

    // parameter and plausibility checks, early returns when failed
    if (mbednn == NULL) return ret;
    if (mbednn->layer_count == 0ul) return ret;
    if ((mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_DENSE) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_INPUT) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_DROPOUT) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_FLATTEN_2D)) return ret;

    // allocate a new layer
    if (mbednn_allocate_layer(mbednn) == MBEDNN_OK)
    {
        // set parameters of the layer
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].type = MBEDNN_LAYER_DENSE;
        strcpy(mbednn->layers[cur_layer].type_name, "Dense");
        mbednn->layers[cur_layer].random_type = random_type;
        mbednn->layers[cur_layer].activation_type = activation_type;

        switch (activation_type)
        {
        case MBEDNN_ACTIVATION_SIGMOID:
            mbednn->layers[cur_layer].activation_func = mbednn_sigmoid_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_sigmoid_derivation;
            break;

        case MBEDNN_ACTIVATION_RELU:
            mbednn->layers[cur_layer].activation_func = mbednn_relu_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_relu_derivation;
            break;

        case MBEDNN_ACTIVATION_LEAKY_RELU:
            mbednn->layers[cur_layer].activation_func = mbednn_leaky_relu_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_leaky_relu_derivation;
            break;

        case MBEDNN_ACTIVATION_TANH:
            mbednn->layers[cur_layer].activation_func = mbednn_tangens_hyperbolicus_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_tangens_hyperbolicus_derivation;
            break;

        case MBEDNN_ACTIVATION_SOFTSIGN:
            mbednn->layers[cur_layer].activation_func = mbednn_softsign_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_softsign_derivation;
            break;

        case MBEDNN_ACTIVATION_SOFTMAX:
            mbednn->layers[cur_layer].activation_func = mbednn_null_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_null_derivation;
            break;

        case MBEDNN_ACTIVATION_NULL:
        default:
            mbednn->layers[cur_layer].activation_func = mbednn_null_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_null_derivation;
            break;
        }

        mbednn->layers[cur_layer].rows = 1ul;
        mbednn->layers[cur_layer].cols = node_count;
        mbednn->layers[cur_layer].node_count = node_count;
        mbednn->layers[cur_layer].depth = 1ul;
        mbednn->layers[cur_layer].filters_count = 0ul;

        // create the matrices of the layer
        mbednn->layers[cur_layer].outputs = (matrix_t**)alloc_aligned_malloc(sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].outputs[0] = matrix_create_zeros(1ul, node_count);
        mbednn->layers[cur_layer].weights = matrix_create_zeros(node_count, mbednn->layers[cur_layer - 1ul].node_count);
        mbednn->layers[cur_layer].biases = matrix_create_zeros(1ul, node_count);
#ifdef MBEDNN_USE_TRAINING
        mbednn->layers[cur_layer].derivatives = (matrix_t**)alloc_aligned_malloc(sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].derivatives[0] = matrix_create_zeros(1ul, node_count);
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)alloc_aligned_malloc(sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].dl_dz[0] = matrix_create_zeros(1ul, node_count);
        mbednn->layers[cur_layer].weights_gradients = matrix_create_zeros(node_count, mbednn->layers[cur_layer - 1ul].node_count);
        mbednn->layers[cur_layer].biases_gradients = matrix_create_zeros(1ul, node_count);
        mbednn->layers[cur_layer].velocities = matrix_create_zeros(node_count, mbednn->layers[cur_layer - 1ul].node_count);
        mbednn->layers[cur_layer].bias_velocities = matrix_create_zeros(1ul, node_count);
        mbednn->layers[cur_layer].momentums = matrix_create_zeros(node_count, mbednn->layers[cur_layer - 1ul].node_count);
        mbednn->layers[cur_layer].bias_momentums = matrix_create_zeros(1ul, node_count);
#else
        mbednn->layers[cur_layer].derivatives = (matrix_t**)NULL;
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)NULL;
        mbednn->layers[cur_layer].weights_gradients = NULL;
        mbednn->layers[cur_layer].biases_gradients = NULL;
        mbednn->layers[cur_layer].velocities = NULL;
        mbednn->layers[cur_layer].bias_velocities = NULL;
        mbednn->layers[cur_layer].momentums = NULL;
        mbednn->layers[cur_layer].bias_momentums = NULL;
#endif

        mbednn->layers[cur_layer].allocated_bytes = (uint32_t)alloc_get_bytes_allocated() - allocated_bytes;
        mbednn->layers[cur_layer].trainable_parameters = mbednn->layers[cur_layer - 1ul].node_count * node_count + node_count;
        ret = cur_layer;
    }
    else
    {
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].allocated_bytes = 0ul;
        mbednn->layers[cur_layer].trainable_parameters = 0ul;
    }

    return ret;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_add_layer_output
**
**     brief    Add an output layer to a neural network. Layer variables are
**              initialized and related matrices are created.
**
**     params   mbednn: pointer to the neural network
**              node_count: number of neurons (nodes) in that layer
**              activation_type: e.g. ReLU, Sigmoid, Tanh, etc.
**              random_type: e.g. uniform, normal, glorot, he, etc.
**     return   layer number if the layer is added successfully, otherwise MBEDNN_NOK
*/
/*---------------------------------------------------------------------------*/
uint32_t mbednn_add_layer_output(mbednn_t *mbednn, uint32_t node_count, mbednn_activation_type activation_type, mbednn_random_type random_type)
{
    uint32_t ret = (uint32_t)MBEDNN_NOK;
    uint32_t cur_layer;
    uint32_t allocated_bytes = (uint32_t)alloc_get_bytes_allocated();

    // parameter and plausibility checks, early returns when failed
    if (mbednn == NULL) return ret;
    if (mbednn->layer_count == 0ul) return ret;
    if ((mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_DENSE) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_INPUT) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_DROPOUT) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_FLATTEN_2D)) return ret;

    // allocate a new layer
    if (mbednn_allocate_layer(mbednn) == MBEDNN_OK)
    {
        // set parameters of the layer
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].type = MBEDNN_LAYER_OUTPUT;
        strcpy(mbednn->layers[cur_layer].type_name, "Output");
        mbednn->layers[cur_layer].random_type = random_type;
        mbednn->layers[cur_layer].activation_type = activation_type;

        switch (activation_type)
        {
        case MBEDNN_ACTIVATION_SIGMOID:
            mbednn->layers[cur_layer].activation_func = mbednn_sigmoid_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_sigmoid_derivation;
            break;

        case MBEDNN_ACTIVATION_RELU:
            mbednn->layers[cur_layer].activation_func = mbednn_relu_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_relu_derivation;
            break;

        case MBEDNN_ACTIVATION_LEAKY_RELU:
            mbednn->layers[cur_layer].activation_func = mbednn_leaky_relu_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_leaky_relu_derivation;
            break;

        case MBEDNN_ACTIVATION_TANH:
            mbednn->layers[cur_layer].activation_func = mbednn_tangens_hyperbolicus_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_tangens_hyperbolicus_derivation;
            break;

        case MBEDNN_ACTIVATION_SOFTSIGN:
            mbednn->layers[cur_layer].activation_func = mbednn_softsign_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_softsign_derivation;
            break;

        case MBEDNN_ACTIVATION_SOFTMAX:
            mbednn->layers[cur_layer].activation_func = mbednn_null_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_null_derivation;
            break;

        case MBEDNN_ACTIVATION_NULL:
        default:
            mbednn->layers[cur_layer].activation_func = mbednn_null_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_null_derivation;
            break;
        }

        mbednn->layers[cur_layer].rows = 1ul;
        mbednn->layers[cur_layer].cols = node_count;
        mbednn->layers[cur_layer].node_count = node_count;
        mbednn->layers[cur_layer].depth = 1ul;
        mbednn->layers[cur_layer].filters_count = 0ul;

        // create the matrices of the layer
        mbednn->layers[cur_layer].outputs = (matrix_t**)alloc_aligned_malloc(sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].outputs[0] = matrix_create_zeros(1ul, node_count);
        mbednn->layers[cur_layer].weights = matrix_create_zeros(node_count, mbednn->layers[cur_layer - 1ul].node_count);
        mbednn->layers[cur_layer].biases = matrix_create_zeros(1ul, node_count);
#ifdef MBEDNN_USE_TRAINING
        mbednn->layers[cur_layer].derivatives = (matrix_t**)alloc_aligned_malloc(sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].derivatives[0] = matrix_create_zeros(1ul, node_count);
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)alloc_aligned_malloc(sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].dl_dz[0] = matrix_create_zeros(1ul, node_count);
        mbednn->layers[cur_layer].weights_gradients = matrix_create_zeros(node_count, mbednn->layers[cur_layer - 1ul].node_count);
        mbednn->layers[cur_layer].biases_gradients = matrix_create_zeros(1ul, node_count);
        mbednn->layers[cur_layer].velocities = matrix_create_zeros(node_count, mbednn->layers[cur_layer - 1ul].node_count);
        mbednn->layers[cur_layer].bias_velocities = matrix_create_zeros(1ul, node_count);
        mbednn->layers[cur_layer].momentums = matrix_create_zeros(node_count, mbednn->layers[cur_layer - 1ul].node_count);
        mbednn->layers[cur_layer].bias_momentums = matrix_create_zeros(1ul, node_count);
#else
        mbednn->layers[cur_layer].derivatives = (matrix_t**)NULL;
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)NULL;
        mbednn->layers[cur_layer].weights_gradients = NULL;
        mbednn->layers[cur_layer].biases_gradients = NULL;
        mbednn->layers[cur_layer].velocities = NULL;
        mbednn->layers[cur_layer].bias_velocities = NULL;
        mbednn->layers[cur_layer].momentums = NULL;
        mbednn->layers[cur_layer].bias_momentums = NULL;
#endif

        mbednn->layers[cur_layer].allocated_bytes = (uint32_t)alloc_get_bytes_allocated() - allocated_bytes;
        mbednn->layers[cur_layer].trainable_parameters = mbednn->layers[cur_layer - 1ul].node_count * node_count + node_count;
        ret = cur_layer;
    }
    else
    {
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].allocated_bytes = 0ul;
        mbednn->layers[cur_layer].trainable_parameters = 0ul;
    }

    return ret;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_add_layer_dropout
**
**     brief    Add a dropout layer to a neural network. Layer variables are
**              initialized and related matrices are created.
**
**     params   mbednn: pointer to the neural network
**              dropout_rate: dropout rate of neurons in that layer
**     return   layer number if the layer is added successfully, otherwise MBEDNN_NOK
*/
/*---------------------------------------------------------------------------*/
uint32_t  mbednn_add_layer_dropout(mbednn_t* mbednn, real dropout_rate)
{
    uint32_t ret = (uint32_t)MBEDNN_NOK;
    uint32_t cur_layer;
    uint32_t node_count;
    uint32_t allocated_bytes = (uint32_t)alloc_get_bytes_allocated();

    // parameter and plausibility checks, early returns when failed
    if (mbednn == NULL) return ret;
    if (mbednn->layer_count == 0ul) return ret;
    if ((dropout_rate < REAL_C(0.0) || dropout_rate >= REAL_C(1.0))) return ret;
    if ((mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_INPUT) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_DENSE) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_FLATTEN_2D)) return ret;

    // allocate a new layer
    if (mbednn_allocate_layer(mbednn) == MBEDNN_OK)
    {
        // set parameters of the layer
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].type = MBEDNN_LAYER_DROPOUT;
        strcpy(mbednn->layers[cur_layer].type_name, "Dropout");
        mbednn->layers[cur_layer].random_type = MBEDNN_RANDOM_NONE;
        mbednn->layers[cur_layer].activation_type = MBEDNN_ACTIVATION_NULL;
        mbednn->layers[cur_layer].activation_func = mbednn_null_activation;
        mbednn->layers[cur_layer].derivation_func = mbednn_null_derivation;
        node_count = mbednn->layers[cur_layer - 1ul].node_count;
        mbednn->layers[cur_layer].rows = 1ul;
        mbednn->layers[cur_layer].cols = node_count;
        mbednn->layers[cur_layer].node_count = node_count;
        mbednn->layers[cur_layer].depth = 1ul;
        mbednn->layers[cur_layer].filters_count = 0ul;
        mbednn->layers[cur_layer].dropout_rate = dropout_rate;

        // create the matrices of the layer
        mbednn->layers[cur_layer].outputs = (matrix_t**)alloc_aligned_malloc(sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].outputs[0] = matrix_create_zeros(1ul, node_count);
#ifdef MBEDNN_USE_TRAINING
        mbednn->layers[cur_layer].derivatives = (matrix_t**)alloc_aligned_malloc(sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].derivatives[0] = matrix_create_zeros(1ul, node_count);
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)alloc_aligned_malloc(sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].dl_dz[0] = matrix_create_zeros(1ul, node_count);

        mbednn->layers[cur_layer].dropout_mask = matrix_create_zeros(1ul, node_count);
        mbednn->layers[cur_layer].input_indices = alloc_aligned_malloc(node_count * sizeof(uint32_t), sizeof(uint32_t));
        for (uint32_t node = 0ul; node < node_count; node++)
        {
            mbednn->layers[cur_layer].input_indices[node] = node;
        }
#else
        mbednn->layers[cur_layer].derivatives = (matrix_t**)NULL;
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)NULL;
        mbednn->layers[cur_layer].dropout_mask = NULL;
        mbednn->layers[cur_layer].input_indices = NULL;
#endif
        // matrices which are not used for current layer type
        mbednn->layers[cur_layer].weights = NULL;
        mbednn->layers[cur_layer].weights_gradients = NULL;
        mbednn->layers[cur_layer].biases = NULL;
        mbednn->layers[cur_layer].biases_gradients = NULL;
        mbednn->layers[cur_layer].velocities = NULL;
        mbednn->layers[cur_layer].bias_velocities = NULL;
        mbednn->layers[cur_layer].momentums = NULL;
        mbednn->layers[cur_layer].bias_momentums = NULL;

        mbednn->layers[cur_layer].allocated_bytes = (uint32_t)alloc_get_bytes_allocated() - allocated_bytes;
        mbednn->layers[cur_layer].trainable_parameters = 0ul;
        ret = cur_layer;
    }
    else
    {
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].allocated_bytes = 0ul;
        mbednn->layers[cur_layer].trainable_parameters = 0ul;
    }

    return ret;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_add_layer_input_2d
**
**     brief    Add a 2D input layer to a neural network. Layer variables are
**              initialized and related matrices are created.
**
**     params   mbednn: pointer to the neural network
**              rows: number of neuron rows in that layer
**              cols: number of neuron columns in that layer
**              depth: depth of that layer
**     return   layer number if the layer is added successfully, otherwise MBEDNN_NOK
*/
/*---------------------------------------------------------------------------*/
uint32_t mbednn_add_layer_input_2d(mbednn_t *mbednn, uint32_t rows, uint32_t cols, uint32_t depth)
{
    uint32_t ret = (uint32_t)MBEDNN_NOK;
    uint32_t cur_layer;
    uint32_t node_count;
    uint32_t allocated_bytes = (uint32_t)alloc_get_bytes_allocated();

    // parameter and plausibility checks, early returns when failed
    if (mbednn == NULL) return ret;
    if (mbednn->layer_count > 0ul) return ret;

    // allocate a new layer
    if (mbednn_allocate_layer(mbednn) == MBEDNN_OK)
    {
        // set parameters of the layer
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].type = MBEDNN_LAYER_INPUT_2D;
        strcpy(mbednn->layers[cur_layer].type_name, "Input2D");
        mbednn->layers[cur_layer].random_type = MBEDNN_RANDOM_NONE;
        mbednn->layers[cur_layer].activation_type = MBEDNN_ACTIVATION_NULL;
        mbednn->layers[cur_layer].activation_func = mbednn_null_activation;
        mbednn->layers[cur_layer].derivation_func = mbednn_null_derivation;
        node_count = rows * cols;
        mbednn->layers[cur_layer].rows = rows;
        mbednn->layers[cur_layer].cols = cols;
        mbednn->layers[cur_layer].node_count = node_count;
        mbednn->layers[cur_layer].depth = depth;
        mbednn->layers[cur_layer].filters_count = 0ul;

        // create the matrices of the layer
        mbednn->layers[cur_layer].outputs = (matrix_t**)alloc_aligned_malloc(depth * sizeof(matrix_t*), 8);
#ifdef MBEDNN_USE_TRAINING
        mbednn->layers[cur_layer].derivatives = (matrix_t**)alloc_aligned_malloc(depth * sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)alloc_aligned_malloc(depth * sizeof(matrix_t*), 8);
#else
        mbednn->layers[cur_layer].derivatives = (matrix_t**)NULL;
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)NULL;
#endif
        for (uint32_t d = 0ul; d < depth; d++)
        {
            mbednn->layers[cur_layer].outputs[d] = matrix_create_zeros(1ul, node_count);
#ifdef MBEDNN_USE_TRAINING
            mbednn->layers[cur_layer].derivatives[d] = matrix_create_zeros(1ul, node_count);
            mbednn->layers[cur_layer].dl_dz[d] = matrix_create_zeros(1ul, node_count);
#endif
        }

        // matrices which are not used for current layer type
        mbednn->layers[cur_layer].weights = NULL;
        mbednn->layers[cur_layer].weights_gradients = NULL;
        mbednn->layers[cur_layer].biases = NULL;
        mbednn->layers[cur_layer].biases_gradients = NULL;
        mbednn->layers[cur_layer].velocities = NULL;
        mbednn->layers[cur_layer].bias_velocities = NULL;
        mbednn->layers[cur_layer].momentums = NULL;
        mbednn->layers[cur_layer].bias_momentums = NULL;

        mbednn->layers[cur_layer].allocated_bytes = (uint32_t)alloc_get_bytes_allocated() - allocated_bytes;
        mbednn->layers[cur_layer].trainable_parameters = 0ul;
        ret = cur_layer;
    }
    else
    {
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].allocated_bytes = 0ul;
        mbednn->layers[cur_layer].trainable_parameters = 0ul;
    }

    return ret;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_add_layer_conv_2d
**
**     brief    Add a 2D convolution layer to a neural network. Layer variables are
**              initialized and related matrices are created.
**
**     params   mbednn: pointer to the neural network
**              filter_numbers: number of filters in that layer
**              filter_rows: number of filter rows
**              filter_cols: number of filter columns
**              filter_stride: filter stepping width 
**              activation_type: e.g. ReLU, Leaky ReLU.
**     return   layer number if the layer is added successfully, otherwise MBEDNN_NOK
*/
/*---------------------------------------------------------------------------*/
uint32_t mbednn_add_layer_conv_2d(mbednn_t *mbednn, uint32_t filter_numbers, uint32_t filter_rows, uint32_t filter_cols, uint32_t filter_stride, mbednn_activation_type activation_type)
{
    uint32_t ret = (uint32_t)MBEDNN_NOK;
    uint32_t cur_layer;
    uint32_t node_count;
    uint32_t depth;
    uint32_t filters_count;
    uint32_t allocated_bytes = (uint32_t)alloc_get_bytes_allocated();

    // parameter and plausibility checks, early returns when failed
    if (mbednn == NULL) return ret;
    if (mbednn->layer_count == 0ul) return ret;
    if ((mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_INPUT_2D) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_MAXPOOL_2D) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_AVRPOOL_2D)) return ret;

    // allocate a new layer
    if (mbednn_allocate_layer(mbednn) == MBEDNN_OK)
    {
        // set parameters of the layer
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].type = MBEDNN_LAYER_CONV_2D;
        strcpy(mbednn->layers[cur_layer].type_name, "Conv2D");
        mbednn->layers[cur_layer].activation_type = activation_type;
        mbednn->layers[cur_layer].random_type = MBEDNN_RANDOM_UNIFORM;

        switch (activation_type)
        {
        case MBEDNN_ACTIVATION_LEAKY_RELU:
            mbednn->layers[cur_layer].activation_func = mbednn_leaky_relu_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_leaky_relu_derivation;
            break;

        case MBEDNN_ACTIVATION_RELU:
        default:
            mbednn->layers[cur_layer].activation_func = mbednn_relu_activation;
            mbednn->layers[cur_layer].derivation_func = mbednn_relu_derivation;
            break;
        }

        depth = filter_numbers;
        filters_count = mbednn->layers[cur_layer - 1ul].depth;
        mbednn->layers[cur_layer].depth = depth;
        mbednn->layers[cur_layer].filters_count = filters_count;
        mbednn->layers[cur_layer].filters_rows = filter_rows;
        mbednn->layers[cur_layer].filters_cols = filter_cols;
        mbednn->layers[cur_layer].filters_stride = filter_stride;
        mbednn->layers[cur_layer].filters_top_padding = (filter_rows - 1ul) / 2ul;
        mbednn->layers[cur_layer].filters_bottom_padding = filter_rows / 2ul;
        mbednn->layers[cur_layer].filters_left_padding = (filter_cols - 1ul) / 2ul;
        mbednn->layers[cur_layer].filters_right_padding = filter_cols / 2ul;

        mbednn->layers[cur_layer].rows = (mbednn->layers[cur_layer - 1ul].rows +
                                          mbednn->layers[cur_layer].filters_top_padding +
                                          mbednn->layers[cur_layer].filters_bottom_padding -
                                          filter_rows) / mbednn->layers[cur_layer].filters_stride + 1ul;
        mbednn->layers[cur_layer].cols = (mbednn->layers[cur_layer - 1ul].cols +
                                          mbednn->layers[cur_layer].filters_left_padding +
                                          mbednn->layers[cur_layer].filters_right_padding -
                                          filter_cols) / mbednn->layers[cur_layer].filters_stride + 1ul;

        node_count = mbednn->layers[cur_layer].rows * mbednn->layers[cur_layer].cols;
        mbednn->layers[cur_layer].node_count = node_count;

        // create the matrices of the layer
        mbednn->layers[cur_layer].outputs = (matrix_t**)alloc_aligned_malloc(depth * sizeof(matrix_t*), 8);
#ifdef MBEDNN_USE_TRAINING
        mbednn->layers[cur_layer].derivatives = (matrix_t**)alloc_aligned_malloc(depth * sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)alloc_aligned_malloc(depth * sizeof(matrix_t*), 8);
#else
        mbednn->layers[cur_layer].derivatives = (matrix_t**)NULL;
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)NULL;
#endif
        for (uint32_t d = 0ul; d < depth; d++)
        {
            mbednn->layers[cur_layer].outputs[d] = matrix_create_zeros(1ul, node_count);
#ifdef MBEDNN_USE_TRAINING
            mbednn->layers[cur_layer].derivatives[d] = matrix_create_zeros(1ul, node_count);
            mbednn->layers[cur_layer].dl_dz[d] = matrix_create_zeros(1ul, node_count);
#endif
        }

        mbednn->layers[cur_layer].filters = (matrix_t***)alloc_aligned_malloc(depth * sizeof(matrix_t**), 8);
        mbednn->layers[cur_layer].filters_biases = matrix_create_zeros(1ul, depth);
#ifdef MBEDNN_USE_TRAINING
        mbednn->layers[cur_layer].filters_gradients = (matrix_t***)alloc_aligned_malloc(depth * sizeof(matrix_t**), 8);
        mbednn->layers[cur_layer].filters_biases_gradients = matrix_create_zeros(1ul, depth);
#else
        mbednn->layers[cur_layer].filters_gradients = (matrix_t***)NULL;
        mbednn->layers[cur_layer].filters_biases_gradients = NULL;
#endif
        mbednn->layers[cur_layer].filters_extern_enable = (bool**)alloc_aligned_malloc(depth * sizeof(bool*), 8);

        for (uint32_t d = 0ul; d < depth; d++)
        {
            mbednn->layers[cur_layer].filters[d] = (matrix_t**)alloc_aligned_malloc(filters_count * sizeof(matrix_t*), 8);
#ifdef MBEDNN_USE_TRAINING
            mbednn->layers[cur_layer].filters_gradients[d] = (matrix_t**)alloc_aligned_malloc(filters_count * sizeof(matrix_t*), 8);
#endif
            mbednn->layers[cur_layer].filters_extern_enable[d] = (bool*)alloc_aligned_malloc(filters_count * sizeof(bool), 8);

            for (uint32_t f = 0ul; f < filters_count; f++)
            {
                mbednn->layers[cur_layer].filters[d][f] = matrix_create_zeros(filter_rows, filter_cols);
#ifdef MBEDNN_USE_TRAINING
                mbednn->layers[cur_layer].filters_gradients[d][f] = matrix_create_zeros(filter_rows, filter_cols);
#endif
                mbednn->layers[cur_layer].filters_extern_enable[d][f] = false;
            }
        }

        // matrices which are not used for current layer type
        mbednn->layers[cur_layer].weights = NULL;
        mbednn->layers[cur_layer].weights_gradients = NULL;
        mbednn->layers[cur_layer].biases = NULL;
        mbednn->layers[cur_layer].biases_gradients = NULL;
        mbednn->layers[cur_layer].velocities = NULL;
        mbednn->layers[cur_layer].bias_velocities = NULL;
        mbednn->layers[cur_layer].momentums = NULL;
        mbednn->layers[cur_layer].bias_momentums = NULL;

        mbednn->layers[cur_layer].allocated_bytes = (uint32_t)alloc_get_bytes_allocated() - allocated_bytes;
        mbednn->layers[cur_layer].trainable_parameters = mbednn->layers[cur_layer].filters_count *
                                                                mbednn->layers[cur_layer].filters_rows *
                                                                mbednn->layers[cur_layer].filters_cols * 
                                                                mbednn->layers[cur_layer].depth + mbednn->layers[cur_layer].depth;
        ret = cur_layer;
    }
    else
    {
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].allocated_bytes = 0ul;
        mbednn->layers[cur_layer].trainable_parameters = 0ul;
    }

    return ret;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_add_layer_maxpooling_2d
**
**     brief    Add a 2D maximum pooling layer to a neural network. Layer variables are
**              initialized and related matrices are created. Only 2x2 window with 
**              stride 2 is supported.
**
**     params   mbednn: pointer to the neural network
**     return   layer number if the layer is added successfully, otherwise MBEDNN_NOK
*/
/*---------------------------------------------------------------------------*/
uint32_t mbednn_add_layer_maxpooling_2d(mbednn_t *mbednn)
{
    uint32_t ret = (uint32_t)MBEDNN_NOK;
    uint32_t cur_layer;
    uint32_t node_count;
    uint32_t depth;
    uint32_t allocated_bytes = (uint32_t)alloc_get_bytes_allocated();

    // parameter and plausibility checks, early returns when failed
    if (mbednn == NULL) return ret;
    if (mbednn->layer_count == 0ul) return ret;
    if ((mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_INPUT_2D) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_MAXPOOL_2D) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_AVRPOOL_2D) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_CONV_2D)) return ret;

    // allocate a new layer
    if (mbednn_allocate_layer(mbednn) == MBEDNN_OK)
    {
        // set parameters of the layer
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].type = MBEDNN_LAYER_MAXPOOL_2D;
        strcpy(mbednn->layers[cur_layer].type_name, "MaxPool2D");
        mbednn->layers[cur_layer].random_type = MBEDNN_RANDOM_NONE;
        mbednn->layers[cur_layer].activation_type = MBEDNN_ACTIVATION_NULL;
        mbednn->layers[cur_layer].activation_func = mbednn_null_activation;
        mbednn->layers[cur_layer].derivation_func = mbednn_null_derivation;
        mbednn->layers[cur_layer].rows = mbednn->layers[cur_layer - 1ul].rows / 2ul;
        mbednn->layers[cur_layer].cols = mbednn->layers[cur_layer - 1ul].cols / 2ul;
        node_count = mbednn->layers[cur_layer].rows * mbednn->layers[cur_layer].cols;
        mbednn->layers[cur_layer].node_count = node_count;
        depth = mbednn->layers[cur_layer - 1ul].depth;
        mbednn->layers[cur_layer].depth = depth;
        mbednn->layers[cur_layer].filters_count = 0ul;

        // create the matrices of the layer
        mbednn->layers[cur_layer].outputs = (matrix_t**)alloc_aligned_malloc(depth * sizeof(matrix_t*), 8);
#ifdef MBEDNN_USE_TRAINING
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)alloc_aligned_malloc(depth * sizeof(matrix_t*), 8);
#else
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)NULL;
#endif
        mbednn->layers[cur_layer].derivatives = (matrix_t**)alloc_aligned_malloc(0, 8);

        for (uint32_t d = 0ul; d < depth; d++)
        {
            mbednn->layers[cur_layer].outputs[d] = matrix_create_zeros(1ul, node_count);
#ifdef MBEDNN_USE_TRAINING
            mbednn->layers[cur_layer].dl_dz[d] = matrix_create_zeros(1ul, node_count);
#endif
        }

        // matrices which are not used for current layer type
        mbednn->layers[cur_layer].weights = NULL;
        mbednn->layers[cur_layer].weights_gradients = NULL;
        mbednn->layers[cur_layer].biases = NULL;
        mbednn->layers[cur_layer].biases_gradients = NULL;
        mbednn->layers[cur_layer].velocities = NULL;
        mbednn->layers[cur_layer].bias_velocities = NULL;
        mbednn->layers[cur_layer].momentums = NULL;
        mbednn->layers[cur_layer].bias_momentums = NULL;

        mbednn->layers[cur_layer].allocated_bytes = (uint32_t)alloc_get_bytes_allocated() - allocated_bytes;
        mbednn->layers[cur_layer].trainable_parameters = 0ul;
        ret = cur_layer;
    }
    else
    {
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].allocated_bytes = 0ul;
        mbednn->layers[cur_layer].trainable_parameters = 0ul;
    }

    return ret;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_add_layer_avrpooling_2d
**
**     brief    Add a 2D average pooling layer to a neural network. Layer variables are
**              initialized and related matrices are created. Only 2x2 window with 
**              stride 2 is supported.
**
**     params   mbednn: pointer to the neural network
**     return   layer number if the layer is added successfully, otherwise MBEDNN_NOK
*/
/*---------------------------------------------------------------------------*/
uint32_t mbednn_add_layer_avrpooling_2d(mbednn_t *mbednn)
{
    uint32_t ret = (uint32_t)MBEDNN_NOK;
    uint32_t cur_layer;
    uint32_t node_count;
    uint32_t depth;
    uint32_t allocated_bytes = (uint32_t)alloc_get_bytes_allocated();

    // parameter and plausibility checks, early returns when failed
    if (mbednn == NULL) return ret;
    if (mbednn->layer_count == 0ul) return ret;
    if ((mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_INPUT_2D) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_MAXPOOL_2D) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_AVRPOOL_2D) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_CONV_2D)) return ret;

    // allocate a new layer
    if (mbednn_allocate_layer(mbednn) == MBEDNN_OK)
    {
        // set parameters of the layer
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].type = MBEDNN_LAYER_AVRPOOL_2D;
        strcpy(mbednn->layers[cur_layer].type_name, "AvrPool2D");
        mbednn->layers[cur_layer].random_type = MBEDNN_RANDOM_NONE;
        mbednn->layers[cur_layer].activation_type = MBEDNN_ACTIVATION_NULL;
        mbednn->layers[cur_layer].activation_func = mbednn_null_activation;
        mbednn->layers[cur_layer].derivation_func = mbednn_null_derivation;
        mbednn->layers[cur_layer].rows = mbednn->layers[cur_layer - 1ul].rows / 2ul;
        mbednn->layers[cur_layer].cols = mbednn->layers[cur_layer - 1ul].cols / 2ul;
        node_count = mbednn->layers[cur_layer].rows * mbednn->layers[cur_layer].cols;
        mbednn->layers[cur_layer].node_count = node_count;
        depth = mbednn->layers[cur_layer - 1ul].depth;
        mbednn->layers[cur_layer].depth = depth;
        mbednn->layers[cur_layer].filters_count = 0ul;

        // create the matrices of the layer
        mbednn->layers[cur_layer].outputs = (matrix_t**)alloc_aligned_malloc(depth * sizeof(matrix_t*), 8);
#ifdef MBEDNN_USE_TRAINING
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)alloc_aligned_malloc(depth * sizeof(matrix_t*), 8);
#else
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)NULL;
#endif
        mbednn->layers[cur_layer].derivatives = (matrix_t**)alloc_aligned_malloc(0, 8);

        for (uint32_t d = 0ul; d < depth; d++)
        {
            mbednn->layers[cur_layer].outputs[d] = matrix_create_zeros(1ul, node_count);
#ifdef MBEDNN_USE_TRAINING
            mbednn->layers[cur_layer].dl_dz[d] = matrix_create_zeros(1ul, node_count);
#endif
        }

        // matrices which are not used for current layer type
        mbednn->layers[cur_layer].weights = NULL;
        mbednn->layers[cur_layer].weights_gradients = NULL;
        mbednn->layers[cur_layer].biases = NULL;
        mbednn->layers[cur_layer].biases_gradients = NULL;
        mbednn->layers[cur_layer].velocities = NULL;
        mbednn->layers[cur_layer].bias_velocities = NULL;
        mbednn->layers[cur_layer].momentums = NULL;
        mbednn->layers[cur_layer].bias_momentums = NULL;

        mbednn->layers[cur_layer].allocated_bytes = (uint32_t)alloc_get_bytes_allocated() - allocated_bytes;
        mbednn->layers[cur_layer].trainable_parameters = 0ul;
        ret = cur_layer;
    }
    else
    {
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].allocated_bytes = 0ul;
        mbednn->layers[cur_layer].trainable_parameters = 0ul;
    }

    return ret;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_add_layer_flatten_2d
**
**     brief    Add a 2D flatten layer to a neural network. Layer variables are
**              initialized and related matrices are created.
**
**     params   mbednn: pointer to the neural network
**     return   layer number if the layer is added successfully, otherwise MBEDNN_NOK
*/
/*---------------------------------------------------------------------------*/
uint32_t mbednn_add_layer_flatten_2d(mbednn_t *mbednn)
{
    uint32_t ret = (uint32_t)MBEDNN_NOK;
    uint32_t cur_layer;
    uint32_t node_count;
    uint32_t allocated_bytes = (uint32_t)alloc_get_bytes_allocated();

    // parameter and plausibility checks, early returns when failed
    if (mbednn == NULL) return ret;
    if (mbednn->layer_count == 0ul) return ret;
    if ((mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_INPUT_2D) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_CONV_2D) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_AVRPOOL_2D) &&
        (mbednn->layers[mbednn->layer_count - 1ul].type != MBEDNN_LAYER_MAXPOOL_2D)) return ret;

    // allocate a new layer
    if (mbednn_allocate_layer(mbednn) == MBEDNN_OK)
    {
        // set parameters of the layer
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].type = MBEDNN_LAYER_FLATTEN_2D;
        strcpy(mbednn->layers[cur_layer].type_name, "Flatten2D");
        mbednn->layers[cur_layer].random_type = MBEDNN_RANDOM_NONE;
        mbednn->layers[cur_layer].activation_type = MBEDNN_ACTIVATION_NULL;
        mbednn->layers[cur_layer].activation_func = mbednn_null_activation;
        mbednn->layers[cur_layer].derivation_func = mbednn_null_derivation;
        node_count = mbednn->layers[cur_layer - 1ul].node_count * mbednn->layers[cur_layer - 1ul].depth;
        mbednn->layers[cur_layer].rows = 1ul;
        mbednn->layers[cur_layer].cols = node_count;
        mbednn->layers[cur_layer].node_count = node_count;
        mbednn->layers[cur_layer].depth = 1ul;
        mbednn->layers[cur_layer].filters_count = 0ul;

        // create the matrices of the layer
        mbednn->layers[cur_layer].outputs = (matrix_t**)alloc_aligned_malloc(sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].outputs[0] = matrix_create_zeros(1ul, node_count);
#ifdef MBEDNN_USE_TRAINING
        mbednn->layers[cur_layer].derivatives = (matrix_t**)alloc_aligned_malloc(sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].derivatives[0] = matrix_create_zeros(1ul, node_count);
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)alloc_aligned_malloc(sizeof(matrix_t*), 8);
        mbednn->layers[cur_layer].dl_dz[0] = matrix_create_zeros(1ul, node_count);
#else
        mbednn->layers[cur_layer].derivatives = (matrix_t**)NULL;
        mbednn->layers[cur_layer].dl_dz = (matrix_t**)NULL;
#endif

        // matrices which are not used for current layer type
        mbednn->layers[cur_layer].weights = NULL;
        mbednn->layers[cur_layer].weights_gradients = NULL;
        mbednn->layers[cur_layer].biases = NULL;
        mbednn->layers[cur_layer].biases_gradients = NULL;
        mbednn->layers[cur_layer].velocities = NULL;
        mbednn->layers[cur_layer].bias_velocities = NULL;
        mbednn->layers[cur_layer].momentums = NULL;
        mbednn->layers[cur_layer].bias_momentums = NULL;

        mbednn->layers[cur_layer].allocated_bytes = (uint32_t)alloc_get_bytes_allocated() - allocated_bytes;
        mbednn->layers[cur_layer].trainable_parameters = 0ul;
        ret = cur_layer;
    }
    else
    {
        cur_layer = mbednn->layer_count - 1ul;
        mbednn->layers[cur_layer].allocated_bytes = 0ul;
        mbednn->layers[cur_layer].trainable_parameters = 0ul;
    }

    return ret;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_free
**
**     brief    Free all resources of a neural network and finally the
**              neural network itself.
**
**     params   mbednn: pointer to the neural network
**     return   void
*/
/*---------------------------------------------------------------------------*/
void mbednn_free(mbednn_t *mbednn)
{
    if (mbednn != NULL)
    {
        for (uint32_t i = 0ul; i < mbednn->layer_count; i++)
        {
            layer_t* layer = &mbednn->layers[i];
            for (uint32_t d = 0ul; d < layer->depth; d++)
            {
                matrix_free_hard(layer->outputs[d]);
#ifdef MBEDNN_USE_TRAINING
                if ((layer->type != MBEDNN_LAYER_AVRPOOL_2D) &&
                    (layer->type != MBEDNN_LAYER_MAXPOOL_2D))
                {
                    matrix_free_hard(layer->derivatives[d]);
                }
                matrix_free_hard(layer->dl_dz[d]);
#endif
            }
            alloc_aligned_free(layer->outputs);
            matrix_free_hard(layer->weights);
            matrix_free_hard(layer->biases);
#ifdef MBEDNN_USE_TRAINING
            alloc_aligned_free(layer->derivatives);
            alloc_aligned_free(layer->dl_dz);
            matrix_free_hard(layer->weights_gradients);
            matrix_free_hard(layer->biases_gradients);
            matrix_free_hard(layer->velocities);
            matrix_free_hard(layer->bias_velocities);
            matrix_free_hard(layer->momentums);
            matrix_free_hard(layer->bias_momentums);
#endif

            if (layer->type == MBEDNN_LAYER_CONV_2D)
            {
                for (uint32_t d = 0ul; d < layer->depth; d++)
                {
                    for (uint32_t f = 0ul; f < layer->filters_count; f++)
                    {
                        matrix_free_hard(layer->filters[d][f]);
#ifdef MBEDNN_USE_TRAINING
                        matrix_free_hard(layer->filters_gradients[d][f]);
#endif
                    }
                    alloc_aligned_free(layer->filters[d]);
                    alloc_aligned_free(layer->filters_extern_enable[d]);
#ifdef MBEDNN_USE_TRAINING
                    alloc_aligned_free(layer->filters_gradients[d]);
#endif
                }
                alloc_aligned_free(layer->filters);
                matrix_free_hard(layer->filters_biases);
                alloc_aligned_free(layer->filters_extern_enable);
#ifdef MBEDNN_USE_TRAINING
                alloc_aligned_free(layer->filters_gradients);
                matrix_free_hard(layer->filters_biases_gradients);
#endif
            }

#ifdef MBEDNN_USE_TRAINING
            if (layer->type == MBEDNN_LAYER_DROPOUT)
            {
                alloc_aligned_free(layer->input_indices);
                alloc_aligned_free(layer->dropout_mask);
            }
#endif
        }
        alloc_aligned_free(mbednn->layers);
        alloc_aligned_free(mbednn);
    }
}

#ifdef MBEDNN_USE_TRAINING
/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_save_binary
**
**     brief    Save the neural network to a binary file.
**
**     params   mbednn: pointer to the neural network
**              filename: the filename
**     return   ok/fail
*/
/*---------------------------------------------------------------------------*/
int32_t mbednn_save_binary(mbednn_t *mbednn, const char *filename)
{
    int32_t ret = MBEDNN_NOK;
    uint32_t elements;
    uint32_t val;
    real val_real;
    FILE *fptr;

    if ((mbednn != NULL) && (filename != NULL))
    {
        fptr = fopen(filename, "wb");
        if (fptr != NULL)
        {
            // save network layer count
            val = mbednn->layer_count;
            fwrite(&val, sizeof(val), 1, fptr);

            // save layer parameters
            for (uint32_t layer = 0ul; layer < mbednn->layer_count; layer++)
            {
                // type
                val = mbednn->layers[layer].type;
                fwrite(&val, sizeof(val), 1, fptr);

                // rows
                val = mbednn->layers[layer].rows;
                fwrite(&val, sizeof(val), 1, fptr);

                // cols
                val = mbednn->layers[layer].cols;
                fwrite(&val, sizeof(val), 1, fptr);

                // nodes
                val = mbednn->layers[layer].node_count;
                fwrite(&val, sizeof(val), 1, fptr);

                // depth
                val = mbednn->layers[layer].depth;
                fwrite(&val, sizeof(val), 1, fptr);

                // filters rows
                val = mbednn->layers[layer].filters_rows;
                fwrite(&val, sizeof(val), 1, fptr);

                // filters cols
                val = mbednn->layers[layer].filters_cols;
                fwrite(&val, sizeof(val), 1, fptr);

                // filters stride
                val = mbednn->layers[layer].filters_stride;
                fwrite(&val, sizeof(val), 1, fptr);

                // random type
                val = mbednn->layers[layer].random_type;
                fwrite(&val, sizeof(val), 1, fptr);

                // activation type
                val = mbednn->layers[layer].activation_type;
                fwrite(&val, sizeof(val), 1, fptr);
            }

            // save optimizer type
            val = mbednn->optimizer_type;
            fwrite(&val, sizeof(val), 1, fptr);

            // save loss type
            val = mbednn->loss_type;
            fwrite(&val, sizeof(val), 1, fptr);

            // save learning rate
            val_real = mbednn->learning_rate;
            fwrite(&val_real, sizeof(val_real), 1, fptr);

            // save epsilon
            val_real = mbednn->epsilon;
            fwrite(&val_real, sizeof(val_real), 1, fptr);

            for (uint32_t layer = 0ul; layer < mbednn->layer_count; layer++)
            {
                switch (mbednn->layers[layer].type)
                {
                case MBEDNN_LAYER_CONV_2D:
                    // save filter biases
                    for (uint32_t n = 0ul; n < mbednn->layers[layer].filters_biases->cols; n++)
                    {
                        val_real = mbednn->layers[layer].filters_biases->values[n];
                        fwrite(&val_real, sizeof(val_real), 1, fptr);
                    }

                    // save filter values
                    elements = mbednn->layers[layer].filters_rows * mbednn->layers[layer].filters_cols;
                    for (uint32_t d = 0ul; d < mbednn->layers[layer].depth; d++)
                    {
                        for (uint32_t f = 0ul; f < mbednn->layers[layer].filters_count; f++)
                        {
                            for (uint32_t n = 0ul; n < elements; n++)
                            {
                                val_real = mbednn->layers[layer].filters[d][f]->values[n];
                                fwrite(&val_real, sizeof(val_real), 1, fptr);
                            }
                        }
                    }
                    break;

                case MBEDNN_LAYER_DENSE:
                case MBEDNN_LAYER_OUTPUT:
                    // save biases
                    for (uint32_t n = 0ul; n < mbednn->layers[layer].biases->cols; n++)
                    {
                        val_real = mbednn->layers[layer].biases->values[n];
                        fwrite(&val_real, sizeof(val_real), 1, fptr);
                    }

                    // save weights
                    elements = mbednn->layers[layer].weights->rows * mbednn->layers[layer].weights->cols;
                    for (uint32_t n = 0ul; n < elements; n++)
                    {
                        val_real = mbednn->layers[layer].weights->values[n];
                        fwrite(&val_real, sizeof(val_real), 1, fptr);
                    }
                    break;

                default:
                    break;
                }
            }

            fclose(fptr);

            ret = MBEDNN_OK;
        }
    }

    return ret;
}
#endif

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_load_binary
**
**     brief    Load a neural network from a binary file.
**
**     params   filename: the filename
**     return   pointer to the created neural network
*/
/*---------------------------------------------------------------------------*/
mbednn_t* mbednn_load_binary(const char *filename)
{
    mbednn_t *mbednn = NULL;
    uint32_t elements;
    uint32_t layer_count;
    uint32_t type;
    uint32_t rows;
    uint32_t cols;
    uint32_t node_count;
    uint32_t depth;
    uint32_t filters_rows;
    uint32_t filters_cols;
    uint32_t filter_stride;
    uint32_t random_type;
    uint32_t activation_type;
    uint32_t optimizer_type;
    uint32_t loss_type;
    real learning_rate;
    real epsilon;
    FILE *fptr;

    if (filename != NULL)
    {
        fptr = fopen(filename, "rb");
        if (fptr != NULL)
        {
            mbednn = mbednn_create();

            if (mbednn != NULL)
            {
                MBEDNN_CHECK_RESULT(fread(&layer_count, sizeof(layer_count), 1, fptr), 1, NULL);

                for (uint32_t layer = 0ul; layer < layer_count; layer++)
                {
                    MBEDNN_CHECK_RESULT(fread(&type, sizeof(type), 1, fptr), 1, NULL);
                    MBEDNN_CHECK_RESULT(fread(&rows, sizeof(rows), 1, fptr), 1, NULL);
                    MBEDNN_CHECK_RESULT(fread(&cols, sizeof(cols), 1, fptr), 1, NULL);
                    MBEDNN_CHECK_RESULT(fread(&node_count, sizeof(node_count), 1, fptr), 1, NULL);
                    MBEDNN_CHECK_RESULT(fread(&depth, sizeof(depth), 1, fptr), 1, NULL);
                    MBEDNN_CHECK_RESULT(fread(&filters_rows, sizeof(filters_rows), 1, fptr), 1, NULL);
                    MBEDNN_CHECK_RESULT(fread(&filters_cols, sizeof(filters_cols), 1, fptr), 1, NULL);
                    MBEDNN_CHECK_RESULT(fread(&filter_stride, sizeof(filter_stride), 1, fptr), 1, NULL);
                    MBEDNN_CHECK_RESULT(fread(&random_type, sizeof(random_type), 1, fptr), 1, NULL);
                    MBEDNN_CHECK_RESULT(fread(&activation_type, sizeof(activation_type), 1, fptr), 1, NULL);

                    switch ((mbednn_layer_type)type)
                    {
                    case MBEDNN_LAYER_INPUT_2D:
                        (void)mbednn_add_layer_input_2d(mbednn, rows, cols, depth);
                        break;
                    case MBEDNN_LAYER_CONV_2D:
                        (void)mbednn_add_layer_conv_2d(mbednn, depth, filters_rows, filters_cols, filter_stride, (mbednn_activation_type)activation_type);
                        break;
                    case MBEDNN_LAYER_MAXPOOL_2D:
                        (void)mbednn_add_layer_maxpooling_2d(mbednn);
                        break;
                    case MBEDNN_LAYER_AVRPOOL_2D:
                        (void)mbednn_add_layer_avrpooling_2d(mbednn);
                        break;
                    case MBEDNN_LAYER_FLATTEN_2D:
                        (void)mbednn_add_layer_flatten_2d(mbednn);
                        break;
                    case MBEDNN_LAYER_INPUT:
                        (void)mbednn_add_layer_input(mbednn, node_count);
                        break;
                    case MBEDNN_LAYER_DENSE:
                        (void)mbednn_add_layer_dense(mbednn, node_count, (mbednn_activation_type)activation_type, (mbednn_random_type)random_type);
                        break;
                    case MBEDNN_LAYER_DROPOUT:
                        (void)mbednn_add_layer_dropout(mbednn, REAL_C(1.0));
                        break;
                    case MBEDNN_LAYER_OUTPUT:
                        (void)mbednn_add_layer_output(mbednn, node_count, (mbednn_activation_type)activation_type, (mbednn_random_type)random_type);
                        break;
                    default:
                        break;
                    }
                }

                // print the network summary
                mbednn_summary(mbednn);

                MBEDNN_CHECK_RESULT(fread(&optimizer_type, sizeof(optimizer_type), 1, fptr), 1, NULL);
                MBEDNN_CHECK_RESULT(fread(&loss_type, sizeof(loss_type), 1, fptr), 1, NULL);
                MBEDNN_CHECK_RESULT(fread(&learning_rate, sizeof(real), 1, fptr), 1, NULL);
                MBEDNN_CHECK_RESULT(fread(&epsilon, sizeof(real), 1, fptr), 1, NULL);

                mbednn_compile(mbednn, optimizer_type, loss_type, learning_rate, epsilon);

                for (uint32_t layer = 0ul; layer < layer_count; layer++)
                {
                    switch (mbednn->layers[layer].type)
                    {
                    case MBEDNN_LAYER_CONV_2D:
                        // load filter biases
                        for (uint32_t n = 0ul; n < mbednn->layers[layer].filters_biases->cols; n++)
                        {
                            MBEDNN_CHECK_RESULT(fread(&mbednn->layers[layer].filters_biases->values[n], sizeof(real), 1, fptr), 1, NULL);
                        }

                        // load filter values
                        elements = mbednn->layers[layer].filters_rows * mbednn->layers[layer].filters_cols;
                        for (uint32_t d = 0ul; d < mbednn->layers[layer].depth; d++)
                        {
                            for (uint32_t f = 0ul; f < mbednn->layers[layer].filters_count; f++)
                            {
                                for (uint32_t n = 0ul; n < elements; n++)
                                {
                                    MBEDNN_CHECK_RESULT(fread(&mbednn->layers[layer].filters[d][f]->values[n], sizeof(real), 1, fptr), 1, NULL);
                                }
                            }
                        }
                        break;

                    case MBEDNN_LAYER_DENSE:
                    case MBEDNN_LAYER_OUTPUT:
                        // load biases
                        for (uint32_t n = 0ul; n < mbednn->layers[layer].biases->cols; n++)
                        {
                            MBEDNN_CHECK_RESULT(fread(&mbednn->layers[layer].biases->values[n], sizeof(real), 1, fptr), 1, NULL);
                        }

                        // load weights
                        elements = mbednn->layers[layer].weights->rows * mbednn->layers[layer].weights->cols;
                        for (uint32_t n = 0ul; n < elements; n++)
                        {
                            MBEDNN_CHECK_RESULT(fread(&mbednn->layers[layer].weights->values[n], sizeof(real), 1, fptr), 1, NULL);
                        }
                        break;

                    default:
                        break;
                    }
                }
            }

            fclose(fptr);
        }
    }

    return mbednn;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_compile
**
**     brief    Set various configuration parameters of a neural network.
**
**     params   mbednn: pointer to the neural network
**              optimizer_type: optimizer function type, e.g. SGD, Adam, RMSprop, Adagrad, etc.
**              loss_type: loss function type, MSE or Categorical Cross-Entropy
**              learning_rate: learning rate
**              epsilon: convergence threshold
**     return   void
*/
/*---------------------------------------------------------------------------*/
void mbednn_compile(mbednn_t *mbednn, mbednn_optimizer_type optimizer_type, mbednn_loss_type loss_type, real learning_rate, real epsilon)
{
    if (mbednn != NULL)
    {
        mbednn->optimizer_type = optimizer_type;

        switch (optimizer_type)
        {
        case MBEDNN_OPTIMIZER_ADAM:
            mbednn->optimization_func = mbednn_optimize_adam;
            mbednn->learning_rate = REAL_C(0.001);
            break;

        case MBEDNN_OPTIMIZER_ADAPT:
            mbednn->optimization_func = mbednn_optimize_sgd;
            mbednn->learning_rate = learning_rate;
            break;

        case MBEDNN_OPTIMIZER_SGD_WITH_DECAY:
            mbednn->optimization_func = mbednn_optimize_sgd;
            mbednn->learning_rate = learning_rate;
            break;

        case MBEDNN_OPTIMIZER_MOMENTUM:
            mbednn->optimization_func = mbednn_optimize_momentum;
            mbednn->learning_rate = REAL_C(0.01);
            break;

        case MBEDNN_OPTIMIZER_RMSPROP:
            mbednn->optimization_func = mbednn_optimize_rmsprop;
            mbednn->learning_rate = REAL_C(0.001);
            break;

        case MBEDNN_OPTIMIZER_ADAGRAD:
            mbednn->optimization_func = mbednn_optimize_adagrad;
            mbednn->learning_rate = REAL_C(0.01);
            break;

        default:
        case MBEDNN_OPTIMIZER_SGD:
            mbednn->optimization_func = mbednn_optimize_sgd;
            mbednn->learning_rate = learning_rate;
            break;
        }

        if (mbednn->layers[mbednn->layer_count - 1ul].activation_type == MBEDNN_ACTIVATION_SOFTMAX)
        {
            // overrule the loss type with cross entropy when output layer has the softmax activation
            // (avoidance of computing the full Jacobian matrix when backpropagating the output layer)
            mbednn->loss_type = MBEDNN_LOSS_CATEGORICAL_CROSS_ENTROPY;
            mbednn->loss_func = mbednn_compute_cross_entropy;
        }
        else
        {
            mbednn->loss_type = loss_type;
            switch (loss_type)
            {
            case MBEDNN_LOSS_CATEGORICAL_CROSS_ENTROPY:
                mbednn->loss_func = mbednn_compute_cross_entropy;
                break;

            default:
            case MBEDNN_LOSS_MSE:
                mbednn->loss_func = mbednn_compute_ms_error;
                break;
            }
        }

        mbednn->epsilon = epsilon;
    }
}

#ifdef MBEDNN_USE_TRAINING
/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_set_filter
**
**     brief    Set all values in a filter.
**
**     params   mbednn: pointer to the neural network
**              layer_index: index of the convolutional layer
**              filter_depth: depth of the filter
**              filter_number: number of the filter
**              values: array of values to set in the filter
**     return   MBEDNN_OK if the filter is set successfully, otherwise MBEDNN_NOK
*/
/*---------------------------------------------------------------------------*/
int32_t mbednn_set_filter(mbednn_t *mbednn, uint32_t layer_index, uint32_t filter_depth, uint32_t filter_number, real *values)
{
    // parameter and plausibility checks, early returns when failed
    if ((mbednn == NULL) || (values == NULL)) return MBEDNN_NOK;
    if (layer_index >= mbednn->layer_count) return MBEDNN_NOK;
    if (mbednn->layers[layer_index].type != MBEDNN_LAYER_CONV_2D) return MBEDNN_NOK;
    if (filter_depth >= mbednn->layers[layer_index].depth) return MBEDNN_NOK;
    if (filter_number >= mbednn->layers[layer_index].filters_count) return MBEDNN_NOK;

    memcpy(mbednn->layers[layer_index].filters[filter_depth][filter_number]->values, values, mbednn->layers[layer_index].filters_rows * mbednn->layers[layer_index].filters_cols * sizeof(real));
    mbednn->layers[layer_index].filters_biases->values[filter_depth] = REAL_C(0.0);
    mbednn->layers[layer_index].filters_extern_enable[filter_depth][filter_number] = true;

    return MBEDNN_OK;
}
#endif

#ifdef MBEDNN_USE_TRAINING
/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_fit
**
**     brief    Training of a neural network with a set of inputs and outputs.
**              Training is done over epochs and batches. Each epoch the
**              training data is randomly shuffled. The training is aborted
**              either when all epochs are processed or when the total loss
**              converged below epsilon.
**
**     params   mbednn: pointer to the neural network
**              inputs: input set training values
**              outputs: expected output set training values
**              epochs: number of total training iterations over the entire dataset
**              batch_size: batch size, typically 32
**     return   total loss of this training
*/
/*---------------------------------------------------------------------------*/
real mbednn_fit(mbednn_t *mbednn, matrix_t *inputs, matrix_t *outputs, uint32_t epochs, uint32_t batch_size)
{
    real loss_per_batch;
    real loss_batch = REAL_C(1.0);
    real loss_per_epoch;
    real loss = REAL_C(1.0);
    uint32_t input_rows;
    uint32_t row;
    uint32_t epoch = 0ul;
    uint32_t batch_count;
    uint32_t *input_indices;
    uint32_t input_node_count;
    uint32_t output_node_count;
    matrix_t **input_set;
    matrix_t *output_set;
    bool converged = false;

    if ((mbednn != NULL) && (inputs != NULL) && (outputs != NULL))
    {
        MBEDNN_PRINTF("Start the training:\n");

        // ensure batch size is appropriate
        input_rows = inputs->rows;
        if (input_rows < 5000ul)
        {
            mbednn->batch_size = 1ul;
        }
        else
        {
            mbednn->batch_size = batch_size;
        }

        // set epoch limit 
        mbednn->epoch_limit = epochs;

        // reset training iteration steps
        mbednn->train_iteration = 0ul;

        // initialize weights to random values
        mbednn_init_weights(mbednn);

        // create indices for shuffling the inputs and outputs
        input_indices = alloc_aligned_malloc(input_rows * sizeof(uint32_t), sizeof(uint32_t));
        for (uint32_t i = 0ul; i < input_rows; i++)
        {
            input_indices[i] = i;
        }

        // get number of input/output nodes
        input_node_count = (mbednn->layers[0ul].node_count);
        output_node_count = (mbednn->layers[mbednn->layer_count - 1ul].node_count);

        // create matrices to hold one single input/output set
        input_set = (matrix_t**)alloc_aligned_malloc(mbednn->layers[0ul].depth * sizeof(matrix_t*), 8);
        for (uint32_t d = 0ul; d < mbednn->layers[0ul].depth; d++)
        {
            input_set[d] = matrix_create(1ul, input_node_count);
        }
        output_set = matrix_create(1ul, output_node_count);

        // calculate number of batches
        batch_count = input_rows / mbednn->batch_size;

        // train over epochs until done
        while (converged == false)
        {
            // re-shuffle the indices for this epoch
            mbednn_shuffle_indices(input_indices, input_rows);

            // iterate over all sets of inputs in this epoch/batch
            epoch ++;

            // reset the loss per epoch
            loss_per_epoch = REAL_C(0.0);

            // iterate over all batches
            for (uint32_t batch = 0ul; batch < batch_count; batch++)
            {
                // zero the gradients
                mbednn_zero_gradients(mbednn);

                // shuffle dropout indices
                mbednn_shuffle_dropout_indices(mbednn);

                // reset the loss per batch
                loss_per_batch = REAL_C(0.0);

                for (uint32_t batch_index = 0ul; batch_index < mbednn->batch_size; batch_index++)
                {
                    row = batch * mbednn->batch_size + batch_index;
                    for (uint32_t d = 0ul; d < mbednn->layers[0ul].depth; d++)
                    {
                        memcpy(input_set[d]->values, inputs->values + input_indices[row] * input_node_count, input_node_count * sizeof(real));
                    }
                    memcpy(output_set->values, outputs->values + input_indices[row] * output_node_count, output_node_count * sizeof(real));

                    // zero the gradients
                    mbednn_zero_dL_dz(mbednn);

                    // train the network with one single dataset
                    loss_per_batch += mbednn_train_pass_network(mbednn, input_set, output_set);

                    // show progress
                    mbednn_print_progress(mbednn->epoch_limit, epoch - 1ul, input_rows, row, loss_batch);
                }

                // average loss over batch size
                loss_per_batch /= (real)mbednn->batch_size;
                loss_batch = loss_per_batch;
                loss_per_epoch += loss_per_batch;

                // update weights based on batched gradients using the chosen optimization function
                mbednn->optimization_func(mbednn);
            }

            // average loss over batch count
            loss = loss_per_epoch / batch_count;
            MBEDNN_PRINTF(" LossPerEpoch=%6.4f", loss);

            // optimize learning once per epoch
            if ((mbednn->optimizer_type == MBEDNN_OPTIMIZER_SGD_WITH_DECAY) || (mbednn->optimizer_type == MBEDNN_OPTIMIZER_MOMENTUM))
            {
                mbednn_optimize_decay(mbednn, loss);
            }
            if (mbednn->optimizer_type == MBEDNN_OPTIMIZER_ADAPT)
            {
                mbednn_optimize_adapt(mbednn, loss);
            }

            if (loss < mbednn->epsilon)
            {
                MBEDNN_PRINTF("\nTraining finished with converging to epsilon.\n");
                converged = true;
            }

            // break when no convergence
            if (epoch >= mbednn->epoch_limit)
            {
                MBEDNN_PRINTF("\nTraining finished with epochs exceeded!\n");
                converged = true;
            }
        }

        // free up matrices
        for (uint32_t d = 0ul; d < mbednn->layers[0ul].depth; d++)
        {
            matrix_free_hard(input_set[d]);
        }

        alloc_aligned_free(input_set);
        matrix_free_hard(output_set);

        // free up indices
        alloc_aligned_free(input_indices);
    }

    return loss;
}
#endif

#ifdef MBEDNN_USE_TRAINING
/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_fit_files
**
**     brief    Training of a neural network with a set of raw data files.
**              Training is done over epochs and batches. Each epoch the
**              training data is randomly shuffled. The training is aborted
**              either when all epochs are processed or when the total loss
**              converged below epsilon.
**
**     params   mbednn: pointer to the neural network
**              epochs: number of total training iterations over the entire dataset
**              batch_size: batch size, typically 32
**              bytes_per_depth: how many bytes one data samples of one channel (depth) has
**              dir_path: path to the directory containing the training raw data files
**              callback: callback function to be called after each epoch
**     return   total loss of this training
*/
/*---------------------------------------------------------------------------*/
real mbednn_fit_files(mbednn_t *mbednn, uint32_t epochs, uint32_t batch_size, uint32_t bytes_per_depth, const char *dir_path, real (*callback)(void))
{
    real loss_per_batch;
    real loss_batch = REAL_C(1.0);
    real loss_per_epoch;
    real loss = REAL_C(1.0);
    real accuracy;
    uint32_t input_rows;
    uint32_t row;
    uint32_t epoch = 0ul;
    uint32_t batch_count;
    uint32_t *input_indices;
    uint32_t input_node_count;
    uint32_t output_node_count;
    matrix_t **input_set;
    matrix_t *output_set;
    bool converged = false;

    if (mbednn != NULL)
    {
        MBEDNN_PRINTF("Start the training:\n");

        // Create the file list
        input_rows = (uint32_t)io_count_files_in_directory(dir_path);
        FileList *file_list = io_create_file_list((int)input_rows);
        io_populate_file_list(file_list, dir_path);

        // ensure batch size is appropriate
        if (input_rows < 5000ul)
        {
            mbednn->batch_size = 1ul;
        }
        else
        {
            mbednn->batch_size = batch_size;
        }

        // set epoch limit 
        mbednn->epoch_limit = epochs;

        // reset training iteration steps
        mbednn->train_iteration = 0ul;

        // initialize weights to random values
        mbednn_init_weights(mbednn);

        // create indices for shuffling the inputs and outputs
        input_indices = alloc_aligned_malloc(input_rows * sizeof(uint32_t), sizeof(uint32_t));
        for (uint32_t i = 0ul; i < input_rows; i++)
        {
            input_indices[i] = i;
        }

        // get number of input/output nodes
        input_node_count = (mbednn->layers[0ul].node_count);
        output_node_count = (mbednn->layers[mbednn->layer_count - 1ul].node_count);

        // create matrices to hold one single input/output set
        input_set = (matrix_t**)alloc_aligned_malloc(mbednn->layers[0ul].depth * sizeof(matrix_t*), 8);
        for (uint32_t d = 0ul; d < mbednn->layers[0ul].depth; d++)
        {
            input_set[d] = matrix_create(1ul, input_node_count);
        }
        output_set = matrix_create(1ul, output_node_count);

        // calculate number of batches
        batch_count = input_rows / mbednn->batch_size;

        // train over epochs until done
        while (converged == false)
        {
            // re-shuffle the indices for this epoch
            mbednn_shuffle_indices(input_indices, input_rows);

            // iterate over all sets of inputs in this epoch/batch
            epoch++;

            // reset the loss per epoch
            loss_per_epoch = REAL_C(0.0);

            // iterate over all batches
            for (uint32_t batch = 0ul; batch < batch_count; batch++)
            {
                // zero the gradients
                mbednn_zero_gradients(mbednn);

                // shuffle dropout indices
                mbednn_shuffle_dropout_indices(mbednn);

                // reset the loss per batch
                loss_per_batch = REAL_C(0.0);

                for (uint32_t batch_index = 0ul; batch_index < mbednn->batch_size; batch_index++)
                {
                    int c;

                    row = batch * mbednn->batch_size + batch_index;

                    c = io_read_file_from_list(file_list, input_indices[row], input_set, mbednn->layers[0ul].depth, bytes_per_depth);
                    matrix_fill_value(output_set, REAL_C(0.0));
                    output_set->values[c] = REAL_C(1.0);

                    // zero the gradients
                    mbednn_zero_dL_dz(mbednn);

                    // train the network with one single dataset
                    loss_per_batch += mbednn_train_pass_network(mbednn, input_set, output_set);

                    // show progress
                    mbednn_print_progress(mbednn->epoch_limit, epoch - 1ul, input_rows, row, loss_batch);
                }

                // average loss over batch size
                loss_per_batch /= (real)mbednn->batch_size;
                loss_batch = loss_per_batch;
                loss_per_epoch += loss_per_batch;

                // update weights based on batched gradients using the chosen optimization function
                mbednn->optimization_func(mbednn);
            }

            // average loss over batch count
            loss = loss_per_epoch / batch_count;
            MBEDNN_PRINTF(" LossPerEpoch=%6.4f", loss);

            // optimize learning once per epoch
            if ((mbednn->optimizer_type == MBEDNN_OPTIMIZER_SGD_WITH_DECAY) || (mbednn->optimizer_type == MBEDNN_OPTIMIZER_MOMENTUM))
            {
                mbednn_optimize_decay(mbednn, loss);
            }
            if (mbednn->optimizer_type == MBEDNN_OPTIMIZER_ADAPT)
            {
                mbednn_optimize_adapt(mbednn, loss);
            }

            if (callback != NULL)
            {
                accuracy = callback();
                if (REAL_FABS(accuracy - REAL_C(100.0)) < REAL_COMP_EPSILON)
                {
                    MBEDNN_PRINTF("\nTraining finished with 100%% accuracy.\n");
                    converged = true;
                }
            }

            if (loss < mbednn->epsilon)
            {
                MBEDNN_PRINTF("\nTraining finished with converging to epsilon.\n");
                converged = true;
            }

            // break when no convergence
            if (epoch >= mbednn->epoch_limit)
            {
                MBEDNN_PRINTF("\nTraining finished with epochs exceeded!\n");
                converged = true;
            }
        }

        // free up matrices
        for (uint32_t d = 0ul; d < mbednn->layers[0ul].depth; d++)
        {
            matrix_free_hard(input_set[d]);
        }

        alloc_aligned_free(input_set);
        matrix_free_hard(output_set);

        // free up file list
        io_free_file_list(file_list);

        // free up indices
        alloc_aligned_free(input_indices);
    }

    return loss;
}
#endif

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_predict
**
**     brief    Predict an output of a single dataset from a trained network.
**
**     params   mbednn: pointer to the neural network
**              inputs: input set test values
**              outputs: predicted output set test values
**     return   MBEDNN_OK if parameters are not NULL, otherwise MBEDNN_NOK
*/
/*---------------------------------------------------------------------------*/
int32_t mbednn_predict(mbednn_t *mbednn, real *inputs, real *outputs)
{
    uint32_t node_count;

    // parameter and plausibility checks, early returns when failed
    if ((mbednn == NULL) || (inputs == NULL) || (outputs == NULL)) return MBEDNN_NOK;

    // set the input nodes for each channel
    node_count = mbednn->layers[0ul].node_count;
    for (uint32_t d = 0ul; d < mbednn->layers[0ul].depth; d++)
    {
        for (uint32_t node = 0ul; node < node_count; node++)
        {
            mbednn->layers[0ul].outputs[d]->values[node] = inputs[node];
        }
    }

    // forward evaluate the network
    mbednn_eval_network(mbednn, false);

    // get the output nodes
    node_count = mbednn->layers[mbednn->layer_count - 1ul].node_count;
    for (uint32_t node = 0ul; node < node_count; node++)
    {
        outputs[node] = mbednn->layers[mbednn->layer_count - 1ul].outputs[0]->values[node];
    }

    return MBEDNN_OK;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_class_predict
**
**     brief    Predict the class from a onehot vector
**
**     params   outputs: the one-hot vector values
**              classes: number of classes of the one-hot vector
**     return   predicted class
*/
/*---------------------------------------------------------------------------*/
uint32_t mbednn_class_predict(real *outputs, uint32_t classes)
{
    uint32_t class = (uint32_t)-1l;
    real prob = REAL_C(-1.0);

    if (outputs != NULL)
    {
        for (uint32_t i = 0ul; i < classes; i++)
        {
            if (outputs[i] > prob)
            {
                prob = outputs[i];
                class = i;
            }
        }
    }

    return class;
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_predict_file
**
**     brief    Predict an output of a single raw data file from a trained network.
**
**     params   mbednn: pointer to the neural network
**              bytes_per_depth: how many bytes one data samples of one channel (depth) has
**              full_path: full path to the raw data file
**              outputs: predicted output set test values
**     return   MBEDNN_OK if parameters are not NULL, otherwise MBEDNN_NOK
*/
/*---------------------------------------------------------------------------*/
int32_t mbednn_predict_file(mbednn_t *mbednn, uint32_t bytes_per_depth, const char *full_path, real *outputs)
{
    uint32_t node_count;

    // parameter and plausibility checks, early returns when failed
    if ((mbednn == NULL) || (outputs == NULL)) return MBEDNN_NOK;

    // set the input nodes for each channel (depth)
    io_read_file(mbednn->layers[0ul].outputs, mbednn->layers[0ul].depth, bytes_per_depth, full_path);

    // forward evaluate the network
    mbednn_eval_network(mbednn, false);

    // get the output nodes
    node_count = mbednn->layers[mbednn->layer_count - 1ul].node_count;
    for (uint32_t node = 0ul; node < node_count; node++)
    {
        outputs[node] = mbednn->layers[mbednn->layer_count - 1ul].outputs[0]->values[node];
    }

    return MBEDNN_OK;
}

#ifdef MBEDNN_USE_TRAINING
/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_get_accuracy
**
**     brief    Get the accuracy of a trained neural network.
**
**     params   mbednn: pointer to the neural network
**              inputs: input set training values
**              outputs: output set training values
**     return   accuracy [0...1]
*/
/*---------------------------------------------------------------------------*/
real mbednn_get_accuracy(mbednn_t *mbednn, matrix_t *inputs, matrix_t *outputs)
{
    real ret = REAL_C(-1.0);
    real* pred;
    uint32_t correct;
    uint32_t classes;
    uint32_t pred_class;
    uint32_t act_class;

    if ((mbednn != NULL) && (inputs != NULL) && (outputs != NULL))
    {
        classes = outputs->cols;
        pred = alloc_aligned_malloc(classes * sizeof(real), sizeof(real));
        correct = 0ul;
        for (uint32_t i = 0ul; i < inputs->rows; i++)
        {
            mbednn_predict(mbednn, &inputs->values[i * inputs->cols], pred);
            pred_class = mbednn_class_predict(pred, classes);
            act_class = mbednn_class_predict(&outputs->values[i * outputs->cols], classes);

            if (pred_class == act_class)
            {
                correct++;
            }
        }
        alloc_aligned_free(pred);

        ret = (real)correct / (real)inputs->rows;
    }

    return ret;
}
#endif

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_print_outputs
**
**     brief    Print nodes of the output layer.
**
**     params   mbednn: pointer to the neural network
**     return   void
*/
/*---------------------------------------------------------------------------*/
void mbednn_print_outputs(mbednn_t *mbednn)
{
    if (mbednn != NULL)
    {
        matrix_print(mbednn->layers[mbednn->layer_count - 1ul].outputs[0]);
    }
}

/*---------------------------------------------------------------------------*/
/*     FUNCTION: mbednn_summary
**
**     brief    Print all network statistics.
**
**     params   mbednn: pointer to the neural network
**     return   void
*/
/*---------------------------------------------------------------------------*/
void mbednn_summary(mbednn_t *mbednn)
{
    uint32_t sum_bytes = 0ul;
    uint32_t sum_nodes = 0ul;
    uint32_t sum_params = 0ul;
    char buf[28];

    MBEDNN_PRINTF("Layer       Type    Bytes    Nodes  Shape(Depth x Rows x Cols)  Filters     TrainP\n");
    MBEDNN_PRINTF("==================================================================================\n");
    for (uint32_t layer = 0ul; layer < mbednn->layer_count; layer++)
    {
        sum_bytes += mbednn->layers[layer].allocated_bytes;
        sum_nodes += mbednn->layers[layer].node_count * mbednn->layers[layer].depth;
        sum_params += mbednn->layers[layer].trainable_parameters;
        sprintf(buf, "(%d x %d x %d)", mbednn->layers[layer].depth,
                         mbednn->layers[layer].rows,
                         mbednn->layers[layer].cols);

        MBEDNN_PRINTF("%5d%11s%9d%9d  %26s     %4d  %9d\n", layer,
                             mbednn->layers[layer].type_name,
                             mbednn->layers[layer].allocated_bytes,
                             mbednn->layers[layer].node_count * mbednn->layers[layer].depth,
                             buf,
                             mbednn->layers[layer].filters_count,
                             mbednn->layers[layer].trainable_parameters);

        MBEDNN_PRINTF("==================================================================================\n");
    }
    MBEDNN_PRINTF("Total sum:      %9d%9d                                       %9d\n\n", sum_bytes, sum_nodes, sum_params);
}
