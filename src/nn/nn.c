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
    F64 bias = 0.1; // small bias to help prevent dead ReLUs
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

// ===================================
// Layer

internal
NN_Layer nn_make_layer_with_random_init(Arena *arena, int input_dim, int output_dim, B32 has_relu) {
    NN_Layer result = {0};
    result.neuron_count = output_dim;
    result.neurons = push_array(arena, NN_Neuron, result.neuron_count);
    for (int i = 0; i < result.neuron_count; ++i) {
        result.neurons[i] = nn_make_neuron_with_random_init(arena, input_dim, has_relu);
    }
    return result;
}

internal
AG_ValueArray nn_layer_apply(Arena *value_arena, Arena *array_arena, NN_Layer *layer, AG_ValueArray x) {
    AG_ValueArray result = {0};

    result.count = layer->neuron_count;
    result.values = push_array(array_arena, AG_Value*, result.count);

    for (int i = 0; i < layer->neuron_count; ++i) {
        result.values[i] = nn_neuron_apply(value_arena, &layer->neurons[i], x);
    }

    return result;
}

internal AG_ValueArray nn_layer_get_params(Arena *arena, NN_Layer *layer) {
    
    ArenaTemp scratch = scratch_begin(&arena, 1);
    
    typedef struct NeuronParamsNode NeuronParamsNode;
    struct NeuronParamsNode {
        NeuronParamsNode *next;
        AG_ValueArray params;
    };

    NeuronParamsNode *first_params = 0, *last_params = 0;
    int total_param_count = 0;
    for (int i = 0; i < layer->neuron_count; ++i) {
        AG_ValueArray neuron_params = nn_neuron_get_params(scratch.arena, &layer->neurons[i]);
        NeuronParamsNode *n = push_array(scratch.arena, NeuronParamsNode, 1);
        n->params = neuron_params;
        SLLQueuePush(first_params, last_params, n);
        total_param_count += neuron_params.count;
    }

    AG_ValueArray result = {0};
    result.count = total_param_count;
    result.values = push_array(arena, AG_Value*, total_param_count);
    int param_idx = 0;
    for (NeuronParamsNode *cur = first_params; cur; cur=cur->next) {
        for (int i = 0; i < cur->params.count; ++i) {
            result.values[param_idx] = cur->params.values[i];
            param_idx += 1;
        }
    }

    scratch_end(scratch);
    return result;
}

// ===================================
// MLP

internal
NN_MLP nn_make_mlp_with_random_init(Arena *arena, int input_dim, int *output_dims, int layer_count) {
    NN_MLP result = {0};
    result.layer_count = layer_count;
    result.layers = push_array(arena, NN_Layer, layer_count);
    for (int i = 0; i < layer_count; ++i) {
        int layer_input_dim = (i > 0 ? output_dims[i-1] : input_dim);
        int layer_output_dim = output_dims[i];
        B32 has_relu = (i != layer_count-1); // output layer has no relu
        result.layers[i] = nn_make_layer_with_random_init(arena, layer_input_dim, layer_output_dim, has_relu);
    }
    return result;
}

internal
AG_ValueArray nn_mlp_get_params(Arena *arena, NN_MLP *mlp) {
    ArenaTemp scratch = scratch_begin(&arena, 1);

    typedef struct LayerParamsNode LayerParamsNode;
    struct LayerParamsNode {
        LayerParamsNode *next;
        AG_ValueArray params;
    };
    LayerParamsNode *first_params = 0, *last_params = 0;
    int total_param_count = 0;
    for (int i = 0; i < mlp->layer_count; ++i) {
        AG_ValueArray lparams = nn_layer_get_params(scratch.arena, &mlp->layers[i]);
        total_param_count += lparams.count;
        LayerParamsNode *n = push_array(scratch.arena, LayerParamsNode, 1);
        n->params = lparams;
        SLLQueuePush(first_params, last_params, n);
    }

    AG_ValueArray result = {0};
    result.count = total_param_count;
    result.values = push_array(arena, AG_Value*, total_param_count);
    int result_idx = 0;
    for (LayerParamsNode *cur = first_params; cur; cur=cur->next) {
        for (int i = 0; i < cur->params.count; ++i) {
            result.values[result_idx] = cur->params.values[i];
            result_idx += 1;
        }
    }
    
    scratch_end(scratch);
    return result;
}

internal
AG_ValueArray nn_mlp_apply(Arena *value_arena, Arena *array_arena, NN_MLP *mlp, AG_ValueArray x) {
    Arena *arenas[] = {value_arena, array_arena};
    ArenaTemp scratch = scratch_begin(arenas, ArrayCount(arenas));
    
    AG_ValueArray current_layer_output = x;
    for (int i = 0; i < mlp->layer_count; ++i) {
        // Put the layer's AG_ValueArray on scratch, since we must *only* put
        // the final AG_ValueArray result on array_arena. (The intermediary 
        // AG_ValueArrays aren't used, nor accessible, by the caller)
        current_layer_output = nn_layer_apply(value_arena, scratch.arena, &mlp->layers[i], current_layer_output);
    }

    // Copy result AG_ValueArray onto array_arena.
    AG_ValueArray result = {0};
    result.count = current_layer_output.count;
    result.values = push_array(array_arena, AG_Value*, result.count);
    ArrayCopy(result.values, current_layer_output.values, result.count);

    scratch_end(scratch);
    return result;
}


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
