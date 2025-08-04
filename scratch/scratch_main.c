#include "../md.c"
#include "../md.h"

#include "../md_alias.h"

typedef struct {
    F64 *data;
    U64 *shape;
    U64  ndims;
} TensorF64;

U64 get_element_count_from_shape(U64 *shape, U64 ndims) {
    if (ndims == 0) return 0;

    U64 result = 1;
    for (int i = 0; i < ndims; ++i) {
        result *= shape[i];
    }

    return result;
}

U64 tensor_get_element_count(TensorF64 *tensor) {
    return get_element_count_from_shape(tensor->shape, tensor->ndims);
}

TensorF64 make_tensor(Arena *arena, F64 *data, U64 *shape, U64 ndims) {
    TensorF64 result = {0};

    U64 element_count = get_element_count_from_shape(shape, ndims);
    result.data = push_array(arena, F64, element_count);
    ArrayCopy(result.data, data, element_count);

    result.shape = push_array(arena, U64, ndims);
    ArrayCopy(result.shape, shape, ndims);
    result.ndims = ndims;

    return result;
}

TensorF64 tensor_add(Arena *arena, TensorF64 *a, TensorF64 *b) {
    Assert(a->ndims == b->ndims);
    B32 shapes_match = 0;
    for (int i = 0; i < MD_Min(a->ndims, b->ndims); ++i) if (a->shape[i]!=b->shape[i]) shapes_match = 0;
    Assert(shapes_match);

    TensorF64 result = {0};

    U64 element_count = tensor_get_element_count(a);
    result.data = push_array(arena, F64, element_count);
    ArrayCopy(result.data, a->data, element_count);

    for (int i = 0; i < element_count; ++i) {
        result.data[i] += b->data[element_count];
    }

    return result;
}

int main(void) {

    Arena *arena = arena_alloc();

    F64 xvals[] = {
        2,    3,  -1,
        3,   -1,   0.5,
        0.5,  1,   1,
        1,    1,  -1
    };
    U64 shape[] = {4, 3};

    TensorF64 x = make_tensor(arena, xvals, shape, ArrayCount(shape));

    F64 yvals[] = {
        1, -1, -1, 1
    };
}