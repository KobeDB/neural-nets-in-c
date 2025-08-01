#include "md.h"
#include "md_alias.h"

#include "autograd.h"

#include <stdio.h>
#include <math.h>

void ag_push_child(Arena *arena, AG_Value *value, AG_Value *child) {
    AG_ChildListNode *node = push_array(arena, AG_ChildListNode, 1);
    node->child = child;
    MD_QueuePush(value->first_child, value->last_child, node);
    value->child_count += 1;
}

AG_Value *ag_leaf(Arena *arena, F64 value) {
    AG_Value *result = push_array(arena, AG_Value, 1);
    result->value = value;
    return result;
}

AG_Value *ag_add(Arena *arena, AG_Value *a, AG_Value *b) {
    AG_Value *result = push_array(arena, AG_Value, 1);

    result->operation = AG_Op_Add;
    result->value = a->value + b->value;

    ag_push_child(arena, result, a);
    ag_push_child(arena, result, b);

    return result;
}

AG_Value *ag_mul(Arena *arena, AG_Value *a, AG_Value *b) {
    AG_Value *result = push_array(arena, AG_Value, 1);

    result->operation = AG_Op_Mul;
    result->value = a->value * b->value;

    ag_push_child(arena, result, a);
    ag_push_child(arena, result, b);

    return result;
}

AG_Value *ag_exp(Arena *arena, AG_Value *x) {
    AG_Value *result = push_array(arena, AG_Value, 1);

    result->operation = AG_Op_Exp;
    result->value = exp(x->value);

    ag_push_child(arena, result, x);

    return result;
}

AG_Value *ag_pow(Arena *arena, AG_Value *a, F64 k) {
    AG_Value *result = push_array(arena, AG_Value, 1);

    result->operation = AG_Op_Pow;
    result->value = exp(k * log(a->value));
    result->k = k;

    ag_push_child(arena, result, a);

    return result;
}

AG_Value *ag_div(Arena *arena, AG_Value *a, AG_Value *b) {
    return ag_mul(arena, a, ag_pow(arena, b, -1));
}

AG_Value *ag_neg(Arena *arena, AG_Value *a) {
    return ag_mul(arena, a, ag_leaf(arena, -1));
}

AG_Value *ag_sub(Arena *arena, AG_Value *a, AG_Value *b) {
    return ag_add(arena, a, ag_neg(arena, b));
}

AG_Value *ag_relu(Arena *arena, AG_Value *a) {
    AG_Value *result = push_array(arena, AG_Value, 1);
    result->operation = AG_Op_Relu;
    result->value = a->value > 0 ? a->value : 0;
    ag_push_child(arena, result, a);
    return result;
}

void ag_build_topo(Arena *arena, AG_Value *value, AG_TopoList *list) {
    if (!value->visited) {
        value->visited = 1;

        for ( AG_ChildListNode *cur = value->first_child; cur != 0; cur = cur->next ) {
            ag_build_topo(arena, cur->child, list);    
        }

        AG_TopoListNode *topo_node = push_array(arena, AG_TopoListNode, 1);
        topo_node->value = value;
        MD_DblPushBack(list->first, list->last, topo_node);
    }
}

void ag_internal_backward(AG_Value *value);

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

void ag_internal_backward(AG_Value *value) {
    
    switch (value->operation) {
        case AG_Op_Null: break;

        case AG_Op_Add: {
            for ( AG_ChildListNode *cur = value->first_child; cur != 0; cur = cur->next ) {
                AG_Value *child = cur->child;
                child->grad += value->grad;
            }
        } break;

        case AG_Op_Mul: {
            F64 product_of_child_values = 1;
            for ( AG_ChildListNode *cur = value->first_child; cur != 0; cur = cur->next ) {
                AG_Value *child = cur->child;
                product_of_child_values *= child->value;
            }

            for ( AG_ChildListNode *cur = value->first_child; cur != 0; cur = cur->next ) {
                AG_Value *child = cur->child;
                child->grad += value->grad * product_of_child_values / child->value;
            }

        } break;

        case AG_Op_Exp: {
            AG_Value *child = value->first_child->child;
            child->grad += value->grad * value->value;
        } break;

        case AG_Op_Pow: {
            AG_Value *child = value->first_child->child;
            F64 k = value->k;
            child->grad += value->grad * k * exp( log(child->value) * (k-1) ); 
        } break;

        case AG_Op_Relu: {
            AG_Value *child = value->first_child->child;
            child->grad += value->grad * (child->value > 0);
        } break;

        default: {
            fprintf(stderr, "Unhandled AG_Op\n");
        } break;
    }

}

typedef struct {
    F64 *weights;
    U64 weight_count;
    F64 bias;
    B32 has_activation;
} AG_Neuron;

AG_Value *ag_neuron(Arena *arena, AG_Neuron *neuron, AG_Value **xs, U64 x_count) {
    MD_Assert(x_count == neuron->weight_count);
    MD_Assert(x_count > 0);

    AG_Value *result = ag_leaf(arena, neuron->bias);

    for (int i = 0; i < x_count; ++i) {
        AG_Value *weighted_x = ag_mul(arena, xs[i], ag_leaf(arena, neuron->weights[i]));
        result = ag_add(arena, result, weighted_x);
    }

    if (neuron->has_activation) result = ag_relu(arena, result);

    return result;
}

void test_nn(void) {
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
        AG_Value *x = ag_leaf(scratch.arena, 10);
        AG_Value *z = ag_pow(scratch.arena, x, 3);
        ag_backward(z);
        printf("");
    }

    {
        AG_Value *x = ag_leaf(scratch.arena, 10);
        AG_Value *y = ag_leaf(scratch.arena, 20);
        AG_Value *z = ag_div(scratch.arena, x, y);
        ag_backward(z);
        printf("");
    }

    {
        AG_Value *x = ag_leaf(scratch.arena, 10);
        AG_Value *z = ag_relu(scratch.arena, x);
        ag_backward(z);

        x = ag_leaf(scratch.arena, 0);
        z = ag_relu(scratch.arena, x);
        ag_backward(z);

        x = ag_leaf(scratch.arena, -10);
        z = ag_relu(scratch.arena, x);
        ag_backward(z);

        printf("");
    }

    {
        AG_Neuron *neuron = push_array(scratch.arena, AG_Neuron, 1);
        neuron->bias = 3.0;
        neuron->weight_count = 5;
        neuron->weights = push_array(scratch.arena, F64, neuron->weight_count);
        neuron->has_activation = 0;
        for (int i = 0; i < neuron->weight_count; ++i) {
            neuron->weights[i] = i * 10;
        }

        AG_Value **xs = push_array(scratch.arena, AG_Value*, neuron->weight_count);
        for (int i = 0; i < neuron->weight_count; ++i) {
            xs[i] = ag_leaf(scratch.arena, i + 1);
        }

        AG_Value *z = ag_neuron(scratch.arena, neuron, xs, neuron->weight_count);

        ag_backward(z);

        printf("");
    }


    scratch_end(scratch);
}