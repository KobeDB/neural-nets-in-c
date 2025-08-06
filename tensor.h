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

// --- Tensor View Creation -----------------------------------------------------------------

Tensor *tensor_make_f64(Arena *arena, F64 *data, U64 element_count, U32 *shape, U32 ndims);

// --- Accessors -----------------------------------------------------------------

void *tensor_get_unchecked(Tensor *tensor, U32 *coords, U32 coord_count);

void *tensor_get(Tensor *tensor, U32 *coords, U32 coord_count);

// --- Tensor Tweaking ----------------------------------------------------------------- 

Tensor *tensor_squeeze(Arena *arena, Tensor *tensor);

// --- Slicing -----------------------------------------------------------------

Tensor *tensor_slice(Arena *arena, Tensor *tensor, RangeU32 *ranges, U32 range_count);

// --- Printing -----------------------------------------------------------------

void tensor_print_f64(FILE *os, Tensor *tensor);
void tensor_print_recursive_f64(FILE *os, Tensor *tensor, U32 dim, U32 *coords, U32 coord_count);

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

#endif