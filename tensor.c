#include "tensor.h"

static void print_coordinates(FILE *os, U32 *coords, U32 coord_count);

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

Tensor *tensor_squeeze(Arena *arena, Tensor *tensor) {
    Tensor *result = push_array(arena, Tensor, 1);

    int new_ndims = 0;
    for (int i = 0; i < tensor->ndims; ++i) {
        if (tensor->shape[i] != 1) new_ndims += 1; 
    }

    result->data = tensor->data;

    result->ndims = new_ndims;
    result->shape = push_array(arena, U32, new_ndims);
    result->strides = push_array(arena, U32, new_ndims);
    
    result->element_size = tensor->element_size;
    result->type_hash = tensor->type_hash;
    result->type_name = tensor->type_name;
    
    int shape_top = 0;
    for (int i = 0; i < tensor->ndims; ++i) {
        if (tensor->shape[i] != 1) {
            result->shape[shape_top]   =  tensor->shape[i];
            result->strides[shape_top] =  tensor->strides[i];
            shape_top += 1;
        }
    }

    return result;
}

static void print_indent(FILE *os, int level) {
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
    fprintf(os, "Tensor { \n");
    fprintf(os, "Shape: ");
    print_coordinates(os, tensor->shape, tensor->ndims);
    fprintf(os, "\n");
    tensor_print_recursive_f64(os, tensor, 0, 0, 0);
    fprintf(os, "\n}");
}

static void print_coordinates(FILE *os, U32 *coords, U32 coord_count) {
    fprintf(os, "[");
    for (int i = 0; i < coord_count; ++i) {
        if (i != 0) fprintf(os, ", ");
        fprintf(os, "%d", coords[i]);
    }
    fprintf(os, "]");
}
