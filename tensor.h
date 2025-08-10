#ifndef TENSOR_H
#define TENSOR_H

#include "md.h"
#include "md.c"

#include "md_alias.h"

// NOTE: Tensors are views; they don't own the data.
//       The element data is owned by an arena (typically).
typedef struct Tensor Tensor;
struct Tensor {
    void *data;

    U32 ndims;
    U32 *shape;
    U32 *strides;

    U64 element_size;
    U64 type_hash;
    String8 type_name;
};

typedef struct {
    U32 start;
    U32 end; // exclusive
} RangeU32;

// --- Tensor (View) Creation -----------------------------------------------------------------

Tensor *tensor_make_view_f64(Arena *arena, F64 *data, U64 element_count, U32 *shape, U32 ndims);
Tensor *tensor_make_view_s32(Arena *arena, S32 *data, U64 element_count, U32 *shape, U32 ndims);

// --- Tensor Cloning -----------------------------------------------------------------

Tensor *tensor_clone(Arena *arena, Tensor *t);

// --- Accessors -----------------------------------------------------------------

void *tensor_get_unchecked(Tensor *tensor, U32 *coords, U32 coord_count);

void *tensor_get(Tensor *tensor, U32 *coords, U32 coord_count);

F64 *tensor_get_f64(Tensor *tensor, U32 *coords, U32 coord_count);

// --- Tensor Tweaking ----------------------------------------------------------------- 

Tensor *tensor_squeeze(Arena *arena, Tensor *tensor);

// --- Slicing -----------------------------------------------------------------

Tensor *tensor_slice(Arena *arena, Tensor *tensor, RangeU32 *ranges, U32 range_count);

// --- Printing -----------------------------------------------------------------

void tensor_print(Tensor *tensor);

void tensor_fprint(FILE *os, Tensor *tensor);

typedef void (TensorElementPrintFunc) (FILE *os, void *element);

void tensor_fprint_custom(FILE *os, Tensor *tensor, TensorElementPrintFunc *print_func);

void tensor_fprint_recursive(FILE *os, Tensor *tensor, U32 dim, U32 *coords, U32 coord_count, TensorElementPrintFunc *print_func);

// --- Element Type Identification ----------------------------------------------------------

// this scheisse was just yoinked from chatgpt. hope it works..
static inline U64 fnv1a_64(const char *str) {
    U64 hash = 14695981039346656037ULL;
    while (*str) {
        hash ^= (U8)(*str++);
        hash *= 1099511628211ULL;
    }
    return hash;
}

#define Tensor_TypeHash(T) fnv1a_64(#T)
#define S8FromType(T) str8_lit(#T)

// --- Arithmetic -----------------------------------------------------------------

Tensor *tensor_add(Arena *arena, Tensor *x, Tensor *y);

typedef void (TensorElementAddFunc) (void *dest, void *to_add);

Tensor *tensor_add_custom(Arena *arena, Tensor *x, Tensor *y, TensorElementAddFunc *add_func);

// --- Helpers -----------------------------------------------------------------

U32 *compute_contiguous_strides(Arena *arena, U32 *shape, U32 ndims);

B32 tensor_shapes_match(Tensor *x, Tensor *y);

#endif