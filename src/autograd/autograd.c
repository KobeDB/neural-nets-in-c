#include <stdio.h>
#include <math.h>

internal
void ag_push_predecessor(Arena *arena, AG_Value *value, AG_Value *pred) {
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

    ag_push_predecessor(arena, result, a);
    ag_push_predecessor(arena, result, b);

    return result;
}

internal
AG_Value *ag_mul(Arena *arena, AG_Value *a, AG_Value *b) {
    AG_Value *result = push_array(arena, AG_Value, 1);

    result->type = AG_ValueType_Mul;
    result->value = a->value * b->value;

    ag_push_predecessor(arena, result, a);
    ag_push_predecessor(arena, result, b);

    return result;
}

internal
AG_Value *ag_exp(Arena *arena, AG_Value *x) {
    AG_Value *result = push_array(arena, AG_Value, 1);

    result->type = AG_ValueType_Exp;
    result->value = exp(x->value);

    ag_push_predecessor(arena, result, x);

    return result;
}

internal
AG_Value *ag_pow(Arena *arena, AG_Value *a, F64 k) {
    AG_Value *result = push_array(arena, AG_Value, 1);

    result->type = AG_ValueType_Pow;
    result->value = pow(a->value, k);
    result->op_params.k = k;

    ag_push_predecessor(arena, result, a);

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
    ag_push_predecessor(arena, result, a);
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

            AG_Value *pred_1 = value->predecessors.first->value;
            AG_Value *pred_2 = value->predecessors.last->value;

            pred_1->grad += value->grad * pred_2->value;
            pred_2->grad += value->grad * pred_1->value;
        } break;

        case AG_ValueType_Exp: {
            AG_Value *pred = value->predecessors.first->value;
            pred->grad += value->grad * value->value;
        } break;

        case AG_ValueType_Pow: {
            AG_Value *pred = value->predecessors.first->value;
            F64 k = value->op_params.k;
            pred->grad += value->grad * k * pow(pred->value, k-1); 
        } break;

        case AG_ValueType_Relu: {
            AG_Value *pred = value->predecessors.first->value;
            pred->grad += value->grad * (pred->value > 0);
        } break;

        default: {
            fprintf(stderr, "ag_internal_backward: unhandled AG_ValueType\n");
        } break;
    }
}

internal
AG_ValueArray ag_value_array_from_raw(Arena *arena, F64 *values, U64 value_count) {
    AG_ValueArray result = {0};
    result.count = value_count;
    result.values = push_array(arena, AG_Value*, result.count);
    for (int i = 0; i < value_count; ++i) {
        result.values[i] = ag_source(arena, values[i]);
    }
    return result;
}

internal
AG_ValueArray ag_make_zero_value_array(Arena *arena, int count) {
    AG_ValueArray result = {0};
    result.count = count;
    result.values = push_array(arena, AG_Value*, result.count);
    for (int i = 0; i < result.count; ++i) {
        result.values[i] = ag_source(arena, 0);
    }
    return result;
}
