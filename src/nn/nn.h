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
    NN_Neuron *neurons;
    int neuron_count;
};

typedef struct NN_MLP NN_MLP;
struct NN_MLP {
    NN_Layer *layers;
    int layer_count;
};

typedef struct NN_ParameterNode NN_ParameterNode;
struct NN_ParameterNode {
    NN_ParameterNode *next;
    AG_Value *value;
};

typedef struct NN_ParameterList NN_ParameterList;
struct NN_ParameterList {
    NN_ParameterNode *first;
    NN_ParameterNode *last;
    int count;
};

// ===================================
// Neuron

internal NN_Neuron nn_make_neuron_from_weights(Arena *arena, F64 *weights, int weight_count, F64 bias, B32 has_relu);

internal NN_Neuron nn_make_neuron_with_random_init(Arena *arena, int input_dim, B32 has_relu);

internal AG_ValueArray nn_neuron_get_params(Arena *arena, NN_Neuron *neuron);

internal AG_Value *nn_neuron_apply(Arena *arena, NN_Neuron *neuron, AG_ValueArray x);

// ===================================
// Layer

internal NN_Layer nn_make_layer_with_random_init(Arena *arena, int input_dim, int output_dim, B32 has_relu);

internal AG_ValueArray nn_layer_apply(Arena *value_arena, Arena *array_arena, NN_Layer *layer,  AG_ValueArray x);

internal AG_ValueArray nn_layer_get_params(Arena *arena, NN_Layer *layer);

// ===================================
// MLP

internal NN_MLP nn_make_mlp_with_random_init(Arena *arena, int input_dim, int *output_dims, int layer_count);

internal
AG_ValueArray nn_mlp_get_params(Arena *arena, NN_MLP *mlp);

internal
AG_ValueArray nn_mlp_apply(Arena *value_arena, Arena *array_arena, NN_MLP *mlp, AG_ValueArray x);

// ============================
// Helpers

internal void push_parameter(Arena *arena, NN_ParameterList *list, AG_Value *param);

internal void append_to_parameter_list(NN_ParameterList *list1, NN_ParameterList *list2);

#endif
