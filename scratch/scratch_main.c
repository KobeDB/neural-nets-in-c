#include "../md.h"
#include "../md.c"

#include "../md_alias.h"

#include <stdio.h>

typedef struct AG_Value AG_Value;

typedef struct {
    int *data;
    int count;
} Temperatures;

typedef struct {
    F64 *data;
    int shape0;
    int shape1;
} HousePrices;

#define Array2DGet(a,x,y) ((a)->data[(x) * (a)->shape1 + (y)])

typedef struct {
    F64 *data;
    int shape0, shape1, shape2;
} Image;

#define Array3DGet(a,x,y,z) (a).data[(x) * (a).shape1 * (a).shape2 + (y) * (a).shape2 + (z)]

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

typedef struct {
    void *data;
    U32 *shape;
    U32 ndims;
    U32 *strides;    
    U64 element_size;
    U64 type_hash;
    String8 type_name;
} Tensor;

void tensor_squeeze(Tensor *tensor) {
    // TODO
}

void *_tensor_get_value(Tensor *tensor, U64 type_hash) {
    if (type_hash != tensor->type_hash) {
        fprintf(stderr, "tensor_get_value: mismatched types\n");
        return 0;
    }

    tensor_squeeze(tensor);
    if (tensor->ndims == 1 && tensor->shape[0] == 1) {
        return tensor->data;
    }

    return 0;
}

#define tensor_get_value(tensor, T) _tensor_get_value(tensor, MD_HashStr(#T)) 

void print_coordinates(FILE *os, U32 *coords, U32 coord_count) {
    fprintf(os, "[");
    for (int i = 0; i < coord_count; ++i) {
        if (i != 0) fprintf(os, ", ");
        fprintf(os, "%d", coords[i]);
    }
    fprintf(os, "]");
}

void *tensor_get_unchecked(Tensor *tensor, U32 *coords, U32 coord_count) {
    
    U64 offset = 0;
    for (int i = 0; i < tensor->ndims; ++i) {
        offset += coords[i] * tensor->strides[i];
    }

    return ((char *)tensor->data) + offset * tensor->element_size;
}

void *tensor_get(Tensor *tensor, U32 *coords, U32 coord_count) {
    if (tensor->ndims != coord_count) {
        fprintf(stderr, "tensor_get: coordinates dimension doesn't match tensor dimension\n");
        return 0;
    }

    for (int i = 0; i < coord_count; ++i) {
        U32 coord = coords[i];
        if (coord >= tensor->shape[i] || coord < 0) {
            fprintf(stderr, "tensor_get: coordinates out of bounds: ");
            print_coordinates(stderr, coords, coord_count);
            fprintf(stderr, " for tensor shape: ");
            print_coordinates(stderr, tensor->shape, tensor->ndims);
            fprintf(stderr, "\n");
            return 0;
        }
    }

    return tensor_get_unchecked(tensor, coords, coord_count);
}

typedef struct {
    U32 start;
    U32 end; // exclusive
} RangeU32;

// This function just creates a view -- not a copy -- of the input tensor. Need to
// find a way to properly document this.
// The passed in arena is just used to allocate the Tensor "view".
Tensor *tensor_slice(Arena *arena, Tensor *tensor, RangeU32 *ranges, U32 range_count) {
    if (tensor->ndims != range_count) {
        fprintf(stderr, "tensor_slice: range count doesn't match tensor dimension\n");
        return 0;
    }

    ArenaTemp scratch = scratch_begin(&arena, 1);

    Tensor *result = push_array(arena, Tensor, 1);

    U32 *first_element_coord = push_array(scratch.arena, U32, range_count);
    for (int i = 0; i < range_count; ++i) {
        first_element_coord[i] = ranges[i].start;
    }

    // TODO: tensor_get error handling
    result->data = tensor_get(tensor, first_element_coord, range_count);

    result->strides = tensor->strides;
    
    result->shape = push_array(arena, U32, tensor->ndims);
    for (int i = 0; i < tensor->ndims; ++i) {
        result->shape[i] = ranges[i].end - ranges[i].start;
    }

    result->element_size = tensor->element_size;
    result->type_hash = tensor->type_hash;
    result->type_name = tensor->type_name;
    result->ndims = tensor->ndims;

    scratch_end(scratch);
    return result;
}

Tensor *tensor_make_f64(Arena *arena, F64 *data, U64 element_count, U32 *shape, U32 ndims) {
    Tensor *result = push_array(arena, Tensor, 1);

    result->type_hash = Tensor_TypeHash(double);
    result->type_name = S8FromType(double);

    result->data = push_array(arena, F64, element_count);
    ArrayCopy(((F64*)result->data), data, element_count);

    result->shape = push_array(arena, U32, ndims);
    ArrayCopy(result->shape, shape, ndims);

    result->strides = push_array(arena, U32, ndims);
    for (int i = 0; i < ndims; ++i) {
        result->strides[i] = 1;
        for (int j = i+1; j < ndims; ++j) {
            result->strides[i] *= shape[j];
        }
    }

    result->ndims = ndims;

    result->element_size = sizeof(double);

    return result;
}

void print_indent(FILE *os, int level) {
    for (int i = 0; i < level; ++i) {
        fprintf(os, "\t");
    }
}

void tensor_print_recursive_f64(FILE *os, Tensor *tensor, U32 dim, U32 *coords, U32 coord_count) {
    ArenaTemp scratch = scratch_begin(0,0);

    print_indent(os, dim);
    fprintf(os, "[");
    if (dim != tensor->ndims-1) fprintf(os, "\n");
    

    for (int i = 0; i < tensor->shape[dim]; ++i) {
        U32 new_coord_count = coord_count+1;
        U32 *new_coords = push_array(scratch.arena, U32, new_coord_count);
        ArrayCopy(new_coords, coords, coord_count);
        new_coords[new_coord_count-1] = i;

        if (dim >= tensor->ndims-1) {
            printf("%f, ", *(F64*)tensor_get(tensor, new_coords, new_coord_count));
        } else {
            tensor_print_recursive_f64(os, tensor, dim+1, new_coords, new_coord_count);
        }
    }

    if (dim != tensor->ndims-1) { 
        fprintf(os, "\n");
        print_indent(os, dim);
    }
    
    fprintf(os, "],");
    fprintf(os, "\n");

    scratch_end(scratch);
}

void tensor_print_f64(FILE *os, Tensor *tensor) {
    tensor_print_recursive_f64(os, tensor, 0, 0, 0);
}

int main(void) {

    printf("Hello from scratch\n");

    Arena *arena = arena_alloc();


    F64 xvals[] = {
        2,    3,  -1,
        3,   -1,   0.69,
        0.42,  1,   1,
        1,    1,  -1
    };
    U32 shape[] = {4, 3};

    Tensor *t = tensor_make_f64(arena, xvals, ArrayCount(xvals), shape, ArrayCount(shape));

    U32 coords[] = {1, 2};
    F64 e0 = *(F64*)tensor_get(t, coords, ArrayCount(coords));

    tensor_print_f64(stdout, t);

    RangeU32 ranges[] = {
        {2,4},
        {1,3}
    };

    Tensor *t2 = tensor_slice(arena, t, ranges, ArrayCount(ranges));

    tensor_print_f64(stdout, t2);

    // HousePrices prices = {0};

    // F64 xvals[] = {
    //     2,    3,  -1,
    //     3,   -1,   0.69,
    //     0.42,  1,   1,
    //     1,    1,  -1
    // };
    // U64 shape[] = {4, 3};

    // prices.data = xvals;
    // prices.shape0 = shape[0];
    // prices.shape1 = shape[1];

    // F64 bruh = Array2DGet(&prices, 1, 2);
    
    // printf("bruh: %f\n", Array2DGet(&prices, 1, 2));

    // Array2DGet(&prices, 1, 2) = 0.69420;

    // printf("bruh: %f\n", Array2DGet(&prices, 1, 2));



}