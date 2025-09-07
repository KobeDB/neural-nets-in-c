#include "random.h"

#include <stdio.h>
#include <math.h>

internal
void ag_push_child(Arena *arena, AG_Value *value, AG_Value *pred) {
    AG_PredecessorNode *node = push_array(arena, AG_PredecessorNode, 1);
    node->value = pred;
    SLLQueuePush(value->predecessors.first, value->predecessors.last, node);
    value->predecessors.count += 1;
}

internal
AG_Value *ag_source(Arena *arena, F64 value) {
    AG_Value *result = push_array(arena, AG_Value, 1);
    result->type = AG_ValueType_Source;
    result->value = value;
    return result;
}

internal
AG_Value *ag_add(Arena *arena, AG_Value *a, AG_Value *b) {
    AG_Value *result = push_array(arena, AG_Value, 1);

    result->type = AG_ValueType_Add;
    result->value = a->value + b->value;

    ag_push_child(arena, result, a);
    ag_push_child(arena, result, b);

    return result;
}

internal
AG_Value *ag_mul(Arena *arena, AG_Value *a, AG_Value *b) {
    AG_Value *result = push_array(arena, AG_Value, 1);

    result->type = AG_ValueType_Mul;
    result->value = a->value * b->value;

    ag_push_child(arena, result, a);
    ag_push_child(arena, result, b);

    return result;
}

internal
AG_Value *ag_exp(Arena *arena, AG_Value *x) {
    AG_Value *result = push_array(arena, AG_Value, 1);

    result->type = AG_ValueType_Exp;
    result->value = exp(x->value);

    ag_push_child(arena, result, x);

    return result;
}

internal
AG_Value *ag_pow(Arena *arena, AG_Value *a, F64 k) {
    AG_Value *result = push_array(arena, AG_Value, 1);

    result->type = AG_ValueType_Pow;
    result->value = pow(a->value, k);
    result->op_params.k = k;

    ag_push_child(arena, result, a);

    return result;
}

internal
AG_Value *ag_div(Arena *arena, AG_Value *a, AG_Value *b) {
    return ag_mul(arena, a, ag_pow(arena, b, -1));
}

internal
AG_Value *ag_neg(Arena *arena, AG_Value *a) {
    return ag_mul(arena, a, ag_source(arena, -1));
}

internal
AG_Value *ag_sub(Arena *arena, AG_Value *a, AG_Value *b) {
    return ag_add(arena, a, ag_neg(arena, b));
}

internal
AG_Value *ag_relu(Arena *arena, AG_Value *a) {
    AG_Value *result = push_array(arena, AG_Value, 1);
    result->type = AG_ValueType_Relu;
    result->value = a->value > 0 ? a->value : 0;
    ag_push_child(arena, result, a);
    return result;
}

internal
void ag_build_topo(Arena *arena, AG_Value *value, AG_TopoList *list) {
    if (!value->visited) {
        value->visited = 1;

        for ( AG_PredecessorNode *cur = value->predecessors.first; cur; cur=cur->next ) {
            ag_build_topo(arena, cur->value, list);    
        }

        AG_TopoListNode *topo_node = push_array(arena, AG_TopoListNode, 1);
        topo_node->value = value;
        DLLPushBack(list->first, list->last, topo_node);
    }
}

internal
void ag_internal_backward(AG_Value *value);

internal
void ag_backward(AG_Value *value) {
    ArenaTemp scratch = scratch_begin(0,0);

    // Build topologically sorted list of the value nodes
    AG_TopoList topo = {0};
    ag_build_topo(scratch.arena, value, &topo);

    value->grad = 1;

    for (AG_TopoListNode *cur = topo.last; cur != 0; cur = cur->prev) {
        ag_internal_backward(cur->value);
        cur->value->visited = 0; // Reset visited flag
    }

    scratch_end(scratch);
}

internal
void ag_internal_backward(AG_Value *value) {
    
    switch (value->type) {
        case AG_ValueType_Null: {
            fprintf(stderr, "ag_internal_backward called on uninitialized value of type AG_ValueType_Null");
        } break;

        case AG_ValueType_Source: break; // nothing to do

        case AG_ValueType_Add: {
            for ( AG_PredecessorNode *cur = value->predecessors.first; cur != 0; cur = cur->next ) {
                AG_Value *pred = cur->value;
                pred->grad += value->grad;
            }
        } break;

        case AG_ValueType_Mul: {
            Assert(value->predecessors.count == 2);

            AG_Value *child_0 = value->predecessors.first->value;
            AG_Value *child_1 = value->predecessors.last->value;

            child_0->grad += value->grad * child_1->value;
            child_1->grad += value->grad * child_0->value;
        } break;

        case AG_ValueType_Exp: {
            AG_Value *child = value->predecessors.first->value;
            child->grad += value->grad * value->value;
        } break;

        case AG_ValueType_Pow: {
            AG_Value *child = value->predecessors.first->value;
            F64 k = value->op_params.k;
            child->grad += value->grad * k * pow(child->value, k-1); 
        } break;

        case AG_ValueType_Relu: {
            AG_Value *child = value->predecessors.first->value;
            child->grad += value->grad * (child->value > 0);
        } break;

        default: {
            fprintf(stderr, "ag_internal_backward: unhandled AG_ValueType\n");
        } break;
    }

}

typedef struct {
    AG_Value **values;
    U64 count;
} AG_ValueList;

typedef struct {
    AG_Value **weights; // TODO: maybe allocate weight Values as a single contiguous array
    U64 weight_count;
    AG_Value *bias;
    B32 has_nonlin_activation;
} AG_Neuron;

AG_Neuron *ag_make_neuron(Arena *arena, U64 input_dim, B32 has_nonlin_activation) {
    AG_Neuron *result = push_array(arena, AG_Neuron, 1);

    result->weights = push_array(arena, AG_Value*, input_dim);
    result->weight_count = input_dim;

    static LCG rng = {0};
    if (!rng.is_seeded) lcg_seed(&rng, 42);

    for (int i = 0; i < input_dim; ++i) {
        F64 r = lcg_next_range_f64(&rng, -1, 1);
        result->weights[i] = ag_source(arena, r);
    }

    result->bias = ag_source(arena, 0);
    result->has_nonlin_activation = has_nonlin_activation;

    return result;
}

AG_ValueList ag_neuron_get_params(Arena *arena, AG_Neuron *neuron) {
    AG_ValueList result = {0};

    result.count = neuron->weight_count + 1; // +1 for the bias
    result.values = push_array(arena, AG_Value*, result.count);
    ArrayCopy(result.values, neuron->weights, neuron->weight_count);
    result.values[neuron->weight_count] = neuron->bias;

    return result;
}

AG_Value *ag_neuron_apply(Arena *arena, AG_Neuron *neuron, AG_ValueList x) {
    MD_Assert(x.count == neuron->weight_count);
    MD_Assert(x.count > 0);

    AG_Value *result = neuron->bias;

    for (int i = 0; i < x.count; ++i) {
        AG_Value *weighted_x = ag_mul(arena, x.values[i], neuron->weights[i]);
        result = ag_add(arena, result, weighted_x);
    }

    if (neuron->has_nonlin_activation) result = ag_relu(arena, result);

    return result;
}

typedef struct {
    AG_Neuron **neurons;
    U64 neuron_count;
} AG_Layer;

AG_Layer *ag_make_layer(Arena *arena, U64 input_dim, U64 output_dim, B32 has_nonlin_activation) {
    AG_Layer *result = push_array(arena, AG_Layer, 1);

    result->neuron_count = output_dim;
    result->neurons = push_array(arena, AG_Neuron*, result->neuron_count);

    for (int i = 0; i < result->neuron_count; ++i) {
        result->neurons[i] = ag_make_neuron(arena, input_dim, has_nonlin_activation);
    }

    return result;
}

U64 ag_layer_get_param_count(AG_Layer *layer) {
    return layer->neuron_count > 0 ? (layer->neurons[0]->weight_count + 1) * layer->neuron_count : 0;
}

AG_ValueList ag_layer_get_params(Arena *arena, AG_Layer *layer) {
    AG_ValueList result = {0};

    ArenaTemp scratch = scratch_begin(&arena, 1);

    if (layer->neuron_count > 0) {
        U64 param_count = ag_layer_get_param_count(layer);
        result.count = param_count;
        result.values = push_array(arena, AG_Value*, result.count);

        // Copy over each neuron's params into result.values
        int param_i = 0;
        for (int neuron_i = 0; neuron_i < layer->neuron_count; ++neuron_i) {
            AG_ValueList neuron_params = ag_neuron_get_params(scratch.arena, layer->neurons[neuron_i]);
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

AG_ValueList ag_layer_apply(Arena *arena, AG_Layer *layer, AG_ValueList x) {
    AG_ValueList result = {0};
    result.values = push_array(arena, AG_Value*, layer->neuron_count);
    result.count = layer->neuron_count;

    for (int i = 0; i < layer->neuron_count; ++i) {
        result.values[i] = ag_neuron_apply(arena, layer->neurons[i], x);
    }

    return result;
}

typedef struct {
    AG_Layer **layers;
    U64 layer_count;
} AG_MLP;

AG_MLP *ag_make_mlp(Arena *arena, U64 input_dim, U64 *layer_output_dims, U64 layer_count) {
    AG_MLP *result = push_array(arena, AG_MLP, 1);

    result->layers = push_array(arena, AG_Layer*, layer_count);
    result->layer_count = layer_count;

    for (int i = 0; i < layer_count; ++i) {
        U64 layer_input_dim = ( i == 0 ? input_dim : layer_output_dims[i-1] );
        B32 nonlin = ( i != layer_count-1 );
        result->layers[i] = ag_make_layer(arena, layer_input_dim, layer_output_dims[i], nonlin);
    }

    return result;
}

AG_ValueList ag_mlp_get_params(Arena *arena, AG_MLP *mlp) {
    AG_ValueList result = {0};

    ArenaTemp scratch = scratch_begin(&arena, 1);

    U64 param_count = 0;
    for (int i = 0; i < mlp->layer_count; ++i) {
        param_count += ag_layer_get_param_count(mlp->layers[i]);
    }

    result.values = push_array(arena, AG_Value*, param_count);
    result.count = param_count;

    int param_i = 0;
    for (int layer_i = 0; layer_i < mlp->layer_count; ++layer_i) {
        AG_ValueList layer_params = ag_layer_get_params(scratch.arena, mlp->layers[layer_i]);
        for (int i = 0; i < layer_params.count; ++i) {
            Assert(param_i < param_count); // Sanity check
            result.values[param_i] = layer_params.values[i];
            param_i += 1;
        }
    }

    scratch_end(scratch);
    return result;
}

AG_ValueList ag_mlp_apply(Arena *arena, AG_MLP *mlp, AG_ValueList x) {
    Assert(mlp->layer_count > 0);
    Assert(mlp->layers[0]->neuron_count > 0);
    Assert(x.count == mlp->layers[0]->neurons[0]->weight_count);

    AG_ValueList result = x;

    for (int i = 0; i < mlp->layer_count; ++i) {
        result = ag_layer_apply(arena, mlp->layers[i], result);
    }

    return result;
}

AG_ValueList ag_make_value_list(Arena *arena, F64 *values, U64 value_count) {
    AG_ValueList result = {0};

    result.count = value_count;
    result.values = push_array(arena, AG_Value*, value_count);

    for (int i = 0; i < value_count; ++i) {
        result.values[i] = ag_source(arena, values[i]);
    }

    return result;
}

void run_neural_network_things(void) {
    ArenaTemp scratch = scratch_begin(0,0);
    
    {
        AG_Value *x = push_array(scratch.arena, AG_Value, 1);
        x->value = 10;

        AG_Value *y = ag_mul(scratch.arena, x, x);
        AG_Value *z = ag_add(scratch.arena, x, y);

        ag_backward(z);

        printf("dz/dx: %f\n", x->grad);
        printf("dz/dy: %f\n", y->grad);
    }

    {
        AG_Value *a = push_array(scratch.arena, AG_Value, 1);
        a->value = 3;
        AG_Value *b = ag_add(scratch.arena, a, a);
        ag_backward(b);
        printf("db/da: %f\n", a->grad);
    }

    {
        AG_Value *x = ag_source(scratch.arena, 10);
        AG_Value *z = ag_pow(scratch.arena, x, 3);
        ag_backward(z);
        printf("");
    }

    {
        AG_Value *x = ag_source(scratch.arena, 10);
        AG_Value *y = ag_source(scratch.arena, 20);
        AG_Value *z = ag_div(scratch.arena, x, y);
        ag_backward(z);
        printf("");
    }

    {
        AG_Value *x = ag_source(scratch.arena, 10);
        AG_Value *z = ag_relu(scratch.arena, x);
        ag_backward(z);

        x = ag_source(scratch.arena, 0);
        z = ag_relu(scratch.arena, x);
        ag_backward(z);

        x = ag_source(scratch.arena, -10);
        z = ag_relu(scratch.arena, x);
        ag_backward(z);

        printf("");
    }

    {
        U64 input_dim = 3;
        AG_MLP *mlp = 0;
        {
            ArenaTemp mlp_making_scratch = scratch_begin(&scratch.arena, 1);
            U64 layer_count = 3;
            U64 *layer_output_dims = push_array(mlp_making_scratch.arena, U64, layer_count);
            layer_output_dims[0] = 4;
            layer_output_dims[1] = 4;
            layer_output_dims[2] = 1;
            mlp = ag_make_mlp(scratch.arena, input_dim, layer_output_dims, layer_count);
            scratch_end(mlp_making_scratch);
        }

        F64 xvals[] = {
            2,    3,  -1,
            3,   -1,   0.5,
            0.5,  1,   1,
            1,    1,  -1
        };
        U64 sample_count = ArrayCount(xvals)/input_dim;
        AG_ValueList *x = push_array(scratch.arena, AG_ValueList, sample_count);
        for (int i = 0; i < sample_count; ++i) 
            x[i] = ag_make_value_list(scratch.arena, &xvals[i * input_dim], input_dim);

        F64 yvals[] = {
            1, -1, -1, 1
        };

        AG_ValueList y = ag_make_value_list(scratch.arena, yvals, ArrayCount(yvals));

        for (int iter = 0; iter < 6; ++iter) {
            ArenaTemp grad_desct_temp = temp_begin(scratch.arena);
            Arena *grad_desct_arena = grad_desct_temp.arena;

            AG_ValueList *y_preds = push_array(grad_desct_arena, AG_ValueList, sample_count);

            for (int i = 0; i < sample_count; ++i) {
                y_preds[i] = ag_mlp_apply(grad_desct_arena, mlp, x[i]);
            }

            printf("y_pred: [");
            for (int i = 0; i < sample_count; ++i) {
                printf(" %f,", y_preds[i].values[0]->value);
            }
            printf("]\n");

            AG_ValueList params = ag_mlp_get_params(grad_desct_arena, mlp);

            for (int i = 0; i < params.count; ++i) {
                params.values[i]->grad = 0;
            }

            AG_Value *loss = ag_source(grad_desct_arena, 0);
            for (int i = 0; i < sample_count; ++i) {
                AG_Value *y_pred = y_preds[i].values[0];
                AG_Value *sample_loss = ag_pow(grad_desct_arena, ag_sub(grad_desct_arena, y_pred, y.values[i]), 2);
                loss = ag_add(grad_desct_arena, loss, sample_loss);
            }

            printf("Loss: %f\n", loss->value);

            ag_backward(loss);

            for (int i = 0; i < params.count; ++i) {
                AG_Value *param = params.values[i];
                param->value -= 0.05 * param->grad;
            }

            temp_end(grad_desct_temp);
        }

        printf("");
    }


    scratch_end(scratch);
}
