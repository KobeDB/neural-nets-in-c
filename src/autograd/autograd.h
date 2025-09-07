#ifndef AUTOGRAD_H
#define AUTOGRAD_H

enum AG_Op {
    AG_Op_Null,
    AG_Op_Add,
    AG_Op_Mul,
    AG_Op_Exp,
    AG_Op_Pow,
    AG_Op_Relu,
};
typedef enum AG_Op AG_Op;

// NOTE: Since a single AG_Value can occur multiple times in a children list,
//       we can't store the links of the child list in AG_Value itself 
//       using sibling pointers and instead need a separate ChildListNode struct.
typedef struct AG_ChildListNode AG_ChildListNode;
struct AG_ChildListNode {
    AG_ChildListNode *next;

    struct AG_Value *child;
};

typedef struct AG_Value AG_Value;
struct AG_Value {
    F64 value;
    AG_Op operation;

    F64 grad;

    AG_ChildListNode *first_child;
    AG_ChildListNode *last_child;
    U64 child_count;

    B32 visited; // Flag used internally by backward pass

    union {
        F64 k; // The exponent of a AG_Op_Pow operation 
    } op_params;
};

//
// Backprop Helper Structs
//

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


#endif
