#include "random.h"

internal
NN_Neuron nn_make_neuron(Arena *arena, U64 input_dim, B32 has_relu, U64 seed) {
    NN_Neuron result = {0};

    result.weights = ag_make_zero_value_array(arena, input_dim);

    static LCG rng = {0};
    if (!rng.is_seeded) lcg_seed(&rng, seed);

    for (int i = 0; i < input_dim; ++i) {
        F64 r = lcg_next_range_f64(&rng, -1, 1);
        result.weights.values[i]->value = r;
    }

    result.bias = ag_source(arena, 0);
    result.has_relu = has_relu;

    return result;
}

internal
AG_ValueArray nn_neuron_get_params(Arena *arena, NN_Neuron *neuron) {
    AG_ValueArray result = {0};
    int weight_count = neuron->weights.count;
    result.count = weight_count + 1; // +1 for the bias
    result.values = push_array(arena, AG_Value*, result.count);
    ArrayCopy(result.values, neuron->weights.values, weight_count);
    result.values[weight_count] = neuron->bias;
    return result;
}

internal
AG_Value *nn_neuron_apply(Arena *arena, NN_Neuron *neuron, AG_ValueArray x) {
    MD_Assert(x.count == neuron->weights.count);
    MD_Assert(x.count > 0);

    AG_Value *result = neuron->bias;

    for (int i = 0; i < x.count; ++i) {
        AG_Value *weighted_x = ag_mul(arena, x.values[i], neuron->weights.values[i]);
        result = ag_add(arena, result, weighted_x);
    }

    if (neuron->has_relu) result = ag_relu(arena, result);

    return result;
}

internal
NN_Layer nn_make_layer(Arena *arena, U64 input_dim, U64 output_dim, B32 has_nonlin_activation, U64 seed) {
    NN_Layer result = {0};

    result.neuron_count = output_dim;
    result.neurons = push_array(arena, NN_Neuron*, result.neuron_count);

    for (int i = 0; i < result.neuron_count; ++i) {
        result.neurons[i] = push_array(arena, NN_Neuron, 1);
        *(result.neurons[i]) = nn_make_neuron(arena, input_dim, has_nonlin_activation, seed);
    }

    return result;
}

internal
int nn_layer_get_param_count(NN_Layer *layer) {
    return layer->neuron_count > 0 ? (layer->neurons[0]->weights.count + 1) * layer->neuron_count : 0;
}

internal
AG_ValueArray nn_layer_get_params(Arena *arena, NN_Layer *layer) {
    AG_ValueArray result = {0};

    ArenaTemp scratch = scratch_begin(&arena, 1);

    if (layer->neuron_count > 0) {
        U64 param_count = nn_layer_get_param_count(layer);
        result.count = param_count;
        result.values = push_array(arena, AG_Value*, result.count);

        // Copy over each neuron's params into result.values
        int param_i = 0;
        for (int neuron_i = 0; neuron_i < layer->neuron_count; ++neuron_i) {
            AG_ValueArray neuron_params = nn_neuron_get_params(scratch.arena, layer->neurons[neuron_i]);
            for (int i = 0; i < neuron_params.count; ++i) {
                Assert(param_i < param_count); // Sanity check
                result.values[param_i] = neuron_params.values[i];
                param_i += 1;
            }
        }
    }

    scratch_end(scratch);
    return result;
}

internal
AG_ValueArray nn_layer_apply(Arena *arena, NN_Layer *layer, AG_ValueArray x) {
    AG_ValueArray result = {0};
    result.values = push_array(arena, AG_Value*, layer->neuron_count);
    result.count = layer->neuron_count;

    for (int i = 0; i < layer->neuron_count; ++i) {
        result.values[i] = nn_neuron_apply(arena, layer->neurons[i], x);
    }

    return result;
}

internal
NN_MLP *nn_make_mlp(Arena *arena, U64 input_dim, U64 *layer_output_dims, U64 layer_count, U64 seed) {
    NN_MLP *result = push_array(arena, NN_MLP, 1);

    result->layers = push_array(arena, NN_Layer*, layer_count);
    result->layer_count = layer_count;

    for (int i = 0; i < layer_count; ++i) {
        U64 layer_input_dim = ( i == 0 ? input_dim : layer_output_dims[i-1] );
        B32 nonlin = ( i != layer_count-1 );
        NN_Layer *layer = push_array(arena, NN_Layer, 1);
        *layer = nn_make_layer(arena, layer_input_dim, layer_output_dims[i], nonlin, seed);
        result->layers[i] = layer;
    }

    return result;
}

internal
AG_ValueArray nn_mlp_get_params(Arena *arena, NN_MLP *mlp) {
    AG_ValueArray result = {0};

    ArenaTemp scratch = scratch_begin(&arena, 1);

    U64 param_count = 0;
    for (int i = 0; i < mlp->layer_count; ++i) {
        param_count += nn_layer_get_param_count(mlp->layers[i]);
    }

    result.values = push_array(arena, AG_Value*, param_count);
    result.count = param_count;

    int param_i = 0;
    for (int layer_i = 0; layer_i < mlp->layer_count; ++layer_i) {
        AG_ValueArray layer_params = nn_layer_get_params(scratch.arena, mlp->layers[layer_i]);
        for (int i = 0; i < layer_params.count; ++i) {
            Assert(param_i < param_count); // Sanity check
            result.values[param_i] = layer_params.values[i];
            param_i += 1;
        }
    }

    scratch_end(scratch);
    return result;
}

internal
AG_ValueArray nn_mlp_apply(Arena *arena, NN_MLP *mlp, AG_ValueArray x) {
    Assert(mlp->layer_count > 0);
    Assert(mlp->layers[0]->neuron_count > 0);
    Assert(x.count == mlp->layers[0]->neurons[0]->weights.count);

    AG_ValueArray result = x;

    for (int i = 0; i < mlp->layer_count; ++i) {
        result = nn_layer_apply(arena, mlp->layers[i], result);
    }

    return result;
}
