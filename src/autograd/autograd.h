#ifndef AUTOGRAD_H
#define AUTOGRAD_H

enum AG_ValueType {
    AG_ValueType_Null,
    AG_ValueType_Source,
    AG_ValueType_Add,
    AG_ValueType_Mul,
    AG_ValueType_Exp,
    AG_ValueType_Pow,
    AG_ValueType_Relu,
};
typedef enum AG_ValueType AG_ValueType;

// NOTE: Since a single AG_Value can occur multiple times in a predecessor list,
//       we can't store the links of the predecessor list in AG_Value itself 
//       using sibling pointers and instead need a separate PredecessorNode struct.
typedef struct AG_PredecessorNode AG_PredecessorNode;
struct AG_PredecessorNode {
    AG_PredecessorNode *next;
    struct AG_Value *value;
};

typedef struct AG_PredecessorList AG_PredecessorList;
struct AG_PredecessorList {
    AG_PredecessorNode *first;
    AG_PredecessorNode *last;
    int count;
};

typedef struct AG_Value AG_Value;
struct AG_Value {
    F64 value;
    AG_ValueType type;

    F64 grad;

    AG_PredecessorList predecessors;

    B32 visited; // Flag used internally by backward pass

    union {
        F64 k; // The exponent of a AG_ValueType_Pow operation 
    } op_params;
};

// ==================================
// Backprop helper structs

typedef struct AG_TopoListNode AG_TopoListNode;
struct AG_TopoListNode {
    AG_TopoListNode *next;
    AG_TopoListNode *prev;
    AG_Value *value;
};

typedef struct AG_TopoList AG_TopoList;
struct AG_TopoList {
    AG_TopoListNode *first;
    AG_TopoListNode *last;
};

// ==================================
// Value construction functions

internal AG_Value *ag_source(Arena *arena, F64 value);

internal AG_Value *ag_add(Arena *arena, AG_Value *a, AG_Value *b);

internal AG_Value *ag_sub(Arena *arena, AG_Value *a, AG_Value *b);

internal AG_Value *ag_mul(Arena *arena, AG_Value *a, AG_Value *b);

internal AG_Value *ag_div(Arena *arena, AG_Value *a, AG_Value *b);

internal AG_Value *ag_neg(Arena *arena, AG_Value *a);

internal AG_Value *ag_relu(Arena *arena, AG_Value *a);

internal AG_Value *ag_exp(Arena *arena, AG_Value *x);

internal AG_Value *ag_pow(Arena *arena, AG_Value *a, F64 k);

// ==================================
// Backprop functions

internal void ag_backward(AG_Value *value);

internal void ag_internal_backward(AG_Value *value);

// ==================================
// Helpers

internal void ag_push_predecessor(Arena *arena, AG_Value *value, AG_Value *pred);

internal void ag_build_topo(Arena *arena, AG_Value *value, AG_TopoList *list);


#endif
