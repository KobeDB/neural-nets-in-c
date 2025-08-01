#include "md.h"
#include "md_alias.h"

#include "autograd.h"

#include <stdio.h>

void ag_push_child(Arena *arena, AG_Value *value, AG_Value *child) {
    AG_ChildListNode *node = push_array(arena, AG_ChildListNode, 1);
    node->child = child;
    MD_QueuePush(value->first_child, value->last_child, node);
    value->child_count += 1;
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

void ag_internal_backward(AG_Value *value);

typedef struct AG_TopoListNode AG_TopoListNode;
struct AG_TopoListNode {
    AG_TopoListNode *next;
    AG_TopoListNode *prev;

    AG_Value *value;
};

typedef struct {
    AG_TopoListNode *first;
    AG_TopoListNode *last;
} AG_TopoList;

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

        default: {
            fprintf(stderr, "Unhandled AG_Op\n");
        } break;
    }

}

void test_nn(void) {
    ArenaTemp scratch = scratch_begin(0,0);
    
    AG_Value *x = push_array(scratch.arena, AG_Value, 1);
    x->value = 10;

    AG_Value *y = ag_mul(scratch.arena, x, x);
    AG_Value *z = ag_add(scratch.arena, x, y);

    ag_backward(z);

    printf("dz/dx: %f\n", x->grad);
    printf("dz/dy: %f\n", y->grad);

    scratch_end(scratch);
}