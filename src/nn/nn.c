#include "random.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

internal
NN_Neuron nn_make_neuron_from_weights(Arena *arena, F64 *weights, int weight_count, F64 bias, B32 has_relu) {
    NN_Neuron result = {0};

    result.weights.count = weight_count;
    result.weights.values = push_array(arena, AG_Value*, weight_count);
    for (int i = 0; i < weight_count; ++i) {
        result.weights.values[i] = ag_source(arena, weights[i]);
    }
    result.bias = ag_source(arena, bias);
    result.has_relu = has_relu;

    return result;
}

internal inline
F64 sample_f64_in_range(F64 min, F64 max) {
    return min + (max-min)*((F64)rand()/(RAND_MAX+1.0));
}

internal
NN_Neuron nn_make_neuron_with_random_init(Arena *arena, int input_dim, B32 has_relu) {
    ArenaTemp scratch = scratch_begin(&arena, 1);
    F64 *weights = push_array(scratch.arena, F64, input_dim);
    F64 kaiming_bound = sqrt(6.0/input_dim);
    local_persist B32 is_seeded = 0;
    if (!is_seeded) srand(time(0)); // TODO: this is pretty bad, needa fix later with proper seeding system to get determinism
    is_seeded = 1;
    for (int i = 0; i < input_dim; ++i) weights[i] = sample_f64_in_range(-kaiming_bound, kaiming_bound);
    F64 bias = 0;
    NN_Neuron result = nn_make_neuron_from_weights(arena, weights, input_dim, bias, has_relu);
    scratch_end(scratch);
    return result;
}

internal
AG_Value *nn_neuron_apply(Arena *arena, NN_Neuron *neuron, AG_ValueArray x) {
    Assert(x.count == neuron->weights.count);
    Assert(x.count > 0);

    AG_Value *neuron_output = neuron->bias;

    for (int i = 0; i < x.count; ++i) {
        AG_Value *weight_node = neuron->weights.values[i];
        AG_Value *weighted_x = ag_mul(arena, x.values[i], weight_node);
        neuron_output = ag_add(arena, neuron_output, weighted_x);
    }

    if (neuron->has_relu) neuron_output = ag_relu(arena, neuron_output);

    return neuron_output;
}

internal
AG_ValueArray nn_neuron_get_params(Arena *arena, NN_Neuron *neuron) {
    AG_ValueArray result = {0};
    int param_count = neuron->weights.count+1; // +1 for bias
    result.values = push_array(arena, AG_Value*, param_count);
    result.count = param_count;
    for (int i = 0; i < neuron->weights.count; ++i) result.values[i] = neuron->weights.values[i];
    result.values[result.count-1] = neuron->bias;
    return result;
}

// internal
// NN_Layer nn_make_layer_with_random_init(Arena *arena, int input_dim, int output_dim, B32 has_relu) {
//     NN_Layer result = {0};
//     result.neuron_count = output_dim;
//     result.neurons = push_array(arena, NN_Neuron, result.neuron_count);
//     for (int i = 0; i < result.neuron_count; ++i) {
//         result.neurons[i] = nn_make_neuron_with_random_init(arena, input_dim, has_relu);
//     }
//     return result;
// }

// internal
// NN_LayerApplyResult nn_layer_apply(Arena *value_arena, Arena *param_arena, NN_Layer *layer, AG_ValueArray x) {
//     NN_LayerApplyResult result = {0};

//     result.layer_outputs.count = layer->neuron_count;
//     result.layer_outputs.values = push_array(param_arena, AG_Value*, result.layer_outputs.count);

//     for (int i = 0; i < layer->neuron_count; ++i) {
//         NN_NeuronApplyResult nresult = nn_neuron_apply(value_arena, param_arena, &layer->neurons[i], x);

//         result.layer_outputs.values[i] = nresult.neuron_output;
        
//         append_to_parameter_list(&result.parameters, &nresult.parameters);
//     }

//     return result;
// }


// internal
// NN_MLP nn_make_mlp_with_random_init(Arena *arena, int input_dim, int *output_dims, int layer_count) {
//     NN_MLP result = {0};
//     result.layer_count = layer_count;
//     result.layers = push_array(arena, NN_Layer, layer_count);
//     for (int i = 0; i < layer_count; ++i) {
//         int layer_input_dim = (i > 0 ? output_dims[i-1] : input_dim);
//         int layer_output_dim = output_dims[i];
//         B32 has_relu = (i != layer_count-1); // output layer has no relu
//         result.layers[i] = nn_make_layer_with_random_init(arena, layer_input_dim, layer_output_dim, has_relu);
//     }
//     return result;
// }

// internal
// NN_MLPApplyResult nn_mlp_apply(Arena *value_arena, Arena *param_arena, NN_MLP *mlp, AG_ValueArray x) {
//     NN_MLPApplyResult result = {0};
//     result.outputs = x;
//     for (int i = 0; i < mlp->layer_count; ++i) {
//         NN_LayerApplyResult lresult = nn_layer_apply(value_arena, param_arena, &mlp->layers[i], result.outputs);
//         result.outputs = lresult.layer_outputs;
//     }
//     return result;
// }

internal
void push_parameter(Arena *arena, NN_ParameterList *list, AG_Value *param) {
    NN_ParameterNode *node = push_array(arena, NN_ParameterNode, 1);
    node->value = param;
    SLLQueuePush(list->first, list->last, node);
    list->count += 1;
}

internal
void append_to_parameter_list(NN_ParameterList *list1, NN_ParameterList *list2) {
    if (!list1->first || !list1->last) {
        *list1 = *list2;
        return;
    }
    list1->last->next = list2->first;
    if (list2->last) list1->last = list2->last;
    list1->count += list2->count;
}

// internal
// NN_MLP *nn_make_mlp(Arena *arena, U64 input_dim, U64 *layer_output_dims, U64 layer_count, U64 seed) {
//     NN_MLP *result = push_array(arena, NN_MLP, 1);

//     result->layers = push_array(arena, NN_Layer*, layer_count);
//     result->layer_count = layer_count;

//     for (int i = 0; i < layer_count; ++i) {
//         U64 layer_input_dim = ( i == 0 ? input_dim : layer_output_dims[i-1] );
//         B32 nonlin = ( i != layer_count-1 );
//         NN_Layer *layer = push_array(arena, NN_Layer, 1);
//         *layer = nn_make_layer(arena, layer_input_dim, layer_output_dims[i], nonlin, seed);
//         result->layers[i] = layer;
//     }

//     return result;
// }

// internal
// AG_ValueArray nn_mlp_get_params(Arena *arena, NN_MLP *mlp) {
//     AG_ValueArray result = {0};

//     ArenaTemp scratch = scratch_begin(&arena, 1);

//     U64 param_count = 0;
//     for (int i = 0; i < mlp->layer_count; ++i) {
//         param_count += nn_layer_get_param_count(mlp->layers[i]);
//     }

//     result.values = push_array(arena, AG_Value*, param_count);
//     result.count = param_count;

//     int param_i = 0;
//     for (int layer_i = 0; layer_i < mlp->layer_count; ++layer_i) {
//         AG_ValueArray layer_params = nn_layer_get_params(scratch.arena, mlp->layers[layer_i]);
//         for (int i = 0; i < layer_params.count; ++i) {
//             Assert(param_i < param_count); // Sanity check
//             result.values[param_i] = layer_params.values[i];
//             param_i += 1;
//         }
//     }

//     scratch_end(scratch);
//     return result;
// }

// internal
// AG_ValueArray nn_mlp_apply(Arena *arena, NN_MLP *mlp, AG_ValueArray x) {
//     Assert(mlp->layer_count > 0);
//     Assert(mlp->layers[0]->neuron_count > 0);
//     Assert(x.count == mlp->layers[0]->neurons[0]->weights.count);

//     AG_ValueArray result = x;

//     for (int i = 0; i < mlp->layer_count; ++i) {
//         result = nn_layer_apply(arena, mlp->layers[i], result);
//     }

//     return result;
// }
