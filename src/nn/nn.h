#ifndef NN_H
#define NN_H

typedef struct NN_Neuron NN_Neuron;
struct NN_Neuron {
    F64 *weights;
    int weight_count;
    F64 bias;
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

typedef struct NN_NeuronApplyResult NN_NeuronApplyResult;
struct NN_NeuronApplyResult {
    AG_Value *neuron_output;
    NN_ParameterList parameters; // list nodes allocated on param_arena
};
// 
// AG_Values are allocated on value_arena
// parameter list is allocated on param_arena
//
internal NN_NeuronApplyResult nn_neuron_apply(Arena *value_arena, Arena *param_arena, NN_Neuron *neuron, AG_ValueArray x);

// ===================================
// Layer

internal NN_Layer nn_make_layer_with_random_init(Arena *arena, int input_dim, int output_dim, B32 has_relu);

typedef struct NN_LayerApplyResult NN_LayerApplyResult;
struct NN_LayerApplyResult {
    AG_ValueArray layer_outputs; // value ptrs array allocated on param_arena
    NN_ParameterList parameters; // list nodes allocated on param_arena
};
internal NN_LayerApplyResult nn_layer_apply(Arena *value_arena, Arena *param_arena, NN_Layer *layer,  AG_ValueArray x);

// ===================================
// MLP

// internal NN_Layer nn_make_layer(Arena *arena, U64 input_dim, U64 output_dim, B32 has_nonlin_activation, U64 seed);

// internal int nn_layer_get_param_count(NN_Layer *layer);

// internal AG_ValueArray nn_layer_get_params(Arena *arena, NN_Layer *layer);

// internal AG_ValueArray nn_layer_apply(Arena *arena, NN_Layer *layer, AG_ValueArray x);

// ============================
// Helpers

internal void push_parameter(Arena *arena, NN_ParameterList *list, AG_Value *param);

internal void append_to_parameter_list(NN_ParameterList *list1, NN_ParameterList *list2);

#endif
