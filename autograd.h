#ifndef AUTOGRAD_H
#define AUTOGRAD_H

typedef enum AG_Op AG_Op;
enum AG_Op {
    AG_Op_Null,
    AG_Op_Add,
    AG_Op_Mul,
};

typedef struct AG_Value AG_Value;

// NOTE: Since a single AG_Value can occur multiple times in a children list,
//       we can't store the links of the child list in AG_Value itself 
//       using sibling pointers and instead need a separate ChildListNode struct.
typedef struct AG_ChildListNode AG_ChildListNode;
struct AG_ChildListNode {
    AG_ChildListNode *next;

    AG_Value *child;
};

typedef struct AG_Value AG_Value;
struct AG_Value {
    F64 value;
    AG_Op operation;

    F64 grad;

    AG_ChildListNode *first_child;
    AG_ChildListNode *last_child;
    U64 child_count;

    B32 visited;
};

#endif