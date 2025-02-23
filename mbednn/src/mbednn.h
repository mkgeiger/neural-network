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

#ifndef __MBEDNN_H
#define __MBEDNN_H

#if defined(__cplusplus)
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

#include "types.h"
#include "matrix.h"

// Matrix Debug printf
#define MBEDNN_PRINTF   printf

// Return Values
#define MBEDNN_NOK   -1
#define MBEDNN_OK     0

// Validation
#define MBEDNN_CHECK_OK(s) if ((s) != MBEDNN_OK) return MBEDNN_NOK
#define MBEDNN_CHECK_RESULT(fn, result, retval) if ((result) != (fn)) return (retval)

// Layer Types
typedef enum {
    MBEDNN_LAYER_INPUT_2D,                        // input layer for a NN
    MBEDNN_LAYER_CONV_2D,                         // convolution layer for a NN
    MBEDNN_LAYER_MAXPOOL_2D,                      // max-pooling layer (window 2x2, stride 2) for a NN
    MBEDNN_LAYER_AVRPOOL_2D,                      // average-pooling layer (window 2x2, stride 2) for a NN
    MBEDNN_LAYER_FLATTEN_2D,                      // input layer for a NN with 2D layers
    MBEDNN_LAYER_INPUT,                           // input layer for a NN without 2D layers
    MBEDNN_LAYER_DENSE,                           // dense layers for a NN
    MBEDNN_LAYER_DROPOUT,                         // dropout layers for a NN
    MBEDNN_LAYER_OUTPUT                           // output layer for a NN
} mbednn_layer_type;

// Activation Function Types
typedef enum {
    MBEDNN_ACTIVATION_NULL,                       // no curve
    MBEDNN_ACTIVATION_SIGMOID,                    // s-shaped curve
    MBEDNN_ACTIVATION_RELU,                       // rectified linear unit curve
    MBEDNN_ACTIVATION_LEAKY_RELU,                 // leaky rectified linear unit curve
    MBEDNN_ACTIVATION_TANH,                       // hyperbolic tangent curve
    MBEDNN_ACTIVATION_SOFTSIGN,                   // softsign curve
    MBEDNN_ACTIVATION_SOFTMAX,                    // softmax function especially for the output 
    MBEDNN_ACTIVATION_DEFAULT = MBEDNN_ACTIVATION_SIGMOID
} mbednn_activation_type;

// Random Function Types
typedef enum {
    MBEDNN_RANDOM_UNIFORM,                        // general purpose, lightweight models
    MBEDNN_RANDOM_NORMAL,                         // general purpose
    MBEDNN_RANDOM_GLOROT_UNIFORM,                 // general purpose, sigmoid, tanh, ReLU activations, classification
    MBEDNN_RANDOM_GLOROT_NORMAL,                  // sensitive training, sigmoid, tanh, or softmax activations, regression
    MBEDNN_RANDOM_HE_UNIFORM,                     // wide Layers, ReLU and variants (Leaky ReLU), classification
    MBEDNN_RANDOM_HE_NORMAL,                      // very deep networks, ReLU and variants (Leaky ReLU), regression
    MBEDNN_RANDOM_ZEROS,                          // rarely used
    MBEDNN_RANDOM_ONES,                           // rarely used
    MBEDNN_RANDOM_NONE,                           // rarely used, same as RANDOM_ZEROS
    MBEDNN_RANDOM_DEFAULT = MBEDNN_RANDOM_UNIFORM
} mbednn_random_type;

// Loss Function Types
typedef enum {
    MBEDNN_LOSS_MSE,                              // mean squared error
    MBEDNN_LOSS_CATEGORICAL_CROSS_ENTROPY,        // categorical cross entropy
    MBEDNN_LOSS_DEFAULT = MBEDNN_LOSS_MSE
} mbednn_loss_type;

// Optimizer Function Types
typedef enum {
    MBEDNN_OPTIMIZER_SGD,                         // stochastic gradient descent
    MBEDNN_OPTIMIZER_SGD_WITH_DECAY,              // stochastic gradient descent with decay
    MBEDNN_OPTIMIZER_ADAPT,                       // stochastic gradient descent with adapt
    MBEDNN_OPTIMIZER_MOMENTUM,                    // stochastic gradient descent with momentum
    MBEDNN_OPTIMIZER_RMSPROP,                     // root mean square propagation
    MBEDNN_OPTIMIZER_ADAGRAD,                     // adaptive gradient algorithm
    MBEDNN_OPTIMIZER_ADAM,                        // adaptive moment estimation
    MBEDNN_OPTIMIZER_DEFAULT = MBEDNN_OPTIMIZER_SGD
} mbednn_optimizer_type;

// Prototypes
typedef struct mbednn_t mbednn_t;
typedef real (*mbednn_activation_func)(real x);
typedef real (*mbednn_derivation_func)(real x);
typedef real (*mbednn_loss_func)(mbednn_t *mbednn, matrix_t *outputs);
typedef void (*mbednn_optimization_func)(mbednn_t *mbednn);

// Layer Type
typedef struct
{
    // Common layer fields
    uint32_t                 node_count;                       // node count
    uint32_t                 rows;                             // rows
    uint32_t                 cols;                             // columns
    uint32_t                 depth;                            // depth
    mbednn_layer_type        type;                             // type
    mbednn_random_type       random_type;                      // random algorithm type
    mbednn_activation_type   activation_type;                  // activation function type
    mbednn_activation_func   activation_func;                  // activation function
    mbednn_derivation_func   derivation_func;                  // derivation function
    matrix_t                 **outputs;                        // matrices of output values
    matrix_t                 **derivatives;                    // matrices of derivatives of the activation function
    matrix_t                 **dl_dz;                          // matrices of loss function gradients (used in output layer for dl_dy)

    // Fully connected layer specific fields
    matrix_t                 *weights;                         // matrix of weights
    matrix_t                 *weights_gradients;               // matrix of gradients for the weights
    matrix_t                 *biases;                          // matrix of biases
    matrix_t                 *biases_gradients;                // matrix of gradients for the biases
    matrix_t                 *velocities;                      // matrix of velocities (gradients)
    matrix_t                 *bias_velocities;                 // matrix of bias velocities (gradients)
    matrix_t                 *momentums;                       // matrix of momentums (gradients)
    matrix_t                 *bias_momentums;                  // matrix of bias momentums (gradients)

    // Convolutional layer specific fields
    uint32_t                 filters_count;                    // filters count
    uint32_t                 filters_rows;                     // filters rows
    uint32_t                 filters_cols;                     // filters columns
    uint32_t                 filters_stride;                   // filters stepping width
    uint32_t                 filters_left_padding;             // filters left padding
    uint32_t                 filters_right_padding;            // filters right padding
    uint32_t                 filters_top_padding;              // filters top padding
    uint32_t                 filters_bottom_padding;           // filters bottom padding
    matrix_t                 ***filters;                       // matrices of 3D-filters
    matrix_t                 ***filters_gradients;             // matrices of 3D-filters gradients
    matrix_t                 *filters_biases;                  // matrix of biases for each 3D-filter
    matrix_t                 *filters_biases_gradients;        // matrix of biases gradients for each 3D-filter
    bool                     **filters_extern_enable;          // filters external configuration

    // Dropout layer specific fields
    real                     dropout_rate;                     // dropout rate
    matrix_t                 *dropout_mask;                    // matrix of the dropout mask
    uint32_t                 *input_indices;                   // random input indices
    
    // Statistic fields
    char                     type_name[10];                    // type name
    uint32_t                 allocated_bytes;                  // allocated bytes by this layer
    uint32_t                 trainable_parameters;             // trainable parameters of this layer
} layer_t;

// Mbed Neural Network Type
typedef struct mbednn_t
{
    layer_t                  *layers;                          // array of layers
    uint32_t                 layer_count;                      // number of layers in the network
    uint32_t                 train_iteration;                  // training iteration step
    uint32_t                 epoch_limit;                      // convergence epoch limit
    uint32_t                 batch_size;                       // size of batches
    uint32_t                 mse_counter;                      // rolling average counter
    real                     learning_rate;                    // learning rate of network
    real                     epsilon;                          // threshold for convergence
    real                     last_mse[MBEDNN_DEFAULT_MSE_AVG]; // for averaging the last n mse values
    mbednn_loss_type         loss_type;                        // type of loss function used
    mbednn_optimizer_type    optimizer_type;                   // type of optimizer function used
    mbednn_loss_func         loss_func;                        // the error function
    mbednn_optimization_func optimization_func;                // learning rate/weight optimizer function
} mbednn_t;

// Creating/Freeing
mbednn_t* mbednn_create(void);
uint32_t  mbednn_add_layer_input(mbednn_t *mbednn, uint32_t node_count);
uint32_t  mbednn_add_layer_dense(mbednn_t *mbednn, uint32_t node_count, mbednn_activation_type activation_type, mbednn_random_type random_type);
uint32_t  mbednn_add_layer_output(mbednn_t *mbednn, uint32_t node_count, mbednn_activation_type activation_type, mbednn_random_type random_type);
uint32_t  mbednn_add_layer_dropout(mbednn_t* mbednn, real dropout_rate);
uint32_t  mbednn_add_layer_input_2d(mbednn_t *mbednn, uint32_t rows, uint32_t cols, uint32_t depth);
uint32_t  mbednn_add_layer_conv_2d(mbednn_t *mbednn, uint32_t filter_numbers, uint32_t filter_rows, uint32_t filter_cols, uint32_t filter_stride, mbednn_activation_type activation_type);
uint32_t  mbednn_add_layer_maxpooling_2d(mbednn_t *mbednn);
uint32_t  mbednn_add_layer_avrpooling_2d(mbednn_t *mbednn);
uint32_t  mbednn_add_layer_flatten_2d(mbednn_t *mbednn);
void      mbednn_free(mbednn_t *mbednn);

// Saving/Loading
#ifdef MBEDNN_USE_TRAINING
int32_t   mbednn_save_binary(mbednn_t *mbednn, const char *filename);
#endif
mbednn_t* mbednn_load_binary(const char *filename);

// Setting Hyper Parameters
void      mbednn_compile(mbednn_t *mbednn, mbednn_optimizer_type optimizer_type, mbednn_loss_type loss_type, real learning_rate, real epsilon);
int32_t   mbednn_set_filter(mbednn_t *mbednn, uint32_t layer_index, uint32_t filter_depth, uint32_t filter_number, real *values);

#ifdef MBEDNN_USE_TRAINING
// Training
real      mbednn_fit(mbednn_t *mbednn, matrix_t *inputs, matrix_t *outputs, uint32_t epochs, uint32_t batch_size);
real      mbednn_fit_files(mbednn_t *mbednn, uint32_t epochs, uint32_t batch_size, uint32_t bytes_per_depth, const char *dir_path, real (*callback)(void));
real      mbednn_get_accuracy(mbednn_t *mbednn, matrix_t *inputs, matrix_t *outputs);
#endif

// Predicting
int32_t   mbednn_predict(mbednn_t *mbednn, real *inputs, real *outputs);
int32_t   mbednn_predict_file(mbednn_t *mbednn, uint32_t bytes_per_depth, const char *full_path, real *outputs);
uint32_t  mbednn_class_predict(real *outputs, uint32_t classes);

// Debugging
void      mbednn_print_outputs(mbednn_t *mbednn);
void      mbednn_summary(mbednn_t *mbednn);

#if defined(__cplusplus)
}
#endif

#endif
