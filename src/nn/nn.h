#ifndef NN_H
#define NN_H

typedef struct NN_Neuron NN_Neuron;
struct NN_Neuron {
    AG_ValueArray weights;
    AG_Value *bias;
    B32 has_relu;
};

typedef struct NN_Layer NN_Layer;
struct NN_Layer {
    NN_Neuron **neurons;
    int neuron_count;
};

typedef struct NN_MLP NN_MLP;
struct NN_MLP {
    NN_Layer **layers;
    int layer_count;
};

// ===============================
// Functions

internal NN_Neuron nn_make_neuron(Arena *arena, U64 input_dim, B32 has_relu, U64 seed);

internal AG_ValueArray nn_neuron_get_params(Arena *arena, NN_Neuron *neuron);

internal AG_Value *nn_neuron_apply(Arena *arena, NN_Neuron *neuron, AG_ValueArray x);


internal NN_Layer nn_make_layer(Arena *arena, U64 input_dim, U64 output_dim, B32 has_nonlin_activation, U64 seed);

internal int nn_layer_get_param_count(NN_Layer *layer);

internal AG_ValueArray nn_layer_get_params(Arena *arena, NN_Layer *layer);

internal AG_ValueArray nn_layer_apply(Arena *arena, NN_Layer *layer, AG_ValueArray x);


#endif
