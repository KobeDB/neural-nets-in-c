#include "tensor.h"

static void print_coordinates(FILE *os, U32 *coords, U32 coord_count);

Tensor *tensor_make_view_f64(Arena *arena, F64 *data, U64 element_count, U32 *shape, U32 ndims) {
    Tensor *result = push_array(arena, Tensor, 1);

    result->data = data;
    
    result->ndims = ndims;
    result->shape = push_array(arena, U32, ndims);
    result->strides = compute_contiguous_strides(arena, shape, ndims);
    
    result->element_size = sizeof(double);
    result->type_hash = Tensor_TypeHash(double);
    result->type_name = S8FromType(double);

    ArrayCopy(result->shape, shape, ndims);

    return result;
}

Tensor *tensor_make_view_s32(Arena *arena, S32 *data, U64 element_count, U32 *shape, U32 ndims) {
    Tensor *result = push_array(arena, Tensor, 1);

    result->data = data;
    
    result->ndims = ndims;
    result->shape = push_array(arena, U32, ndims);
    result->strides = compute_contiguous_strides(arena, shape, ndims);
    
    result->element_size = sizeof(uint32_t);
    result->type_hash = Tensor_TypeHash(uint32_t);
    result->type_name = S8FromType(uint32_t);

    ArrayCopy(result->shape, shape, ndims);

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

F64 *tensor_get_f64(Tensor *tensor, U32 *coords, U32 coord_count) {
    if (tensor->type_hash != Tensor_TypeHash(double)) {
        return 0;
    }
    return tensor_get(tensor, coords, coord_count);
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


void tensor_fprint_recursive(FILE *os, Tensor *tensor, U32 dim, U32 *coords, U32 coord_count, TensorElementPrintFunc *print_func) {
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
            print_func(os, tensor_get(tensor, new_coords, new_coord_count));
            fprintf(os, ", ");
        } else {
            tensor_fprint_recursive(os, tensor, dim+1, new_coords, new_coord_count, print_func);
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

void tensor_fprint_custom(FILE *os, Tensor *tensor, TensorElementPrintFunc *print_func) {
    fprintf(os, "Tensor { \n");
    fprintf(os, "Shape: ");
    print_coordinates(os, tensor->shape, tensor->ndims);
    fprintf(os, "\n");
    tensor_fprint_recursive(os, tensor, 0, 0, 0, print_func);
    fprintf(os, "\n}");
}

static inline void tensor_element_print_func_f64(FILE *os, void *element) {
    fprintf(os, "%f", *(F64*)element);
}

static inline void tensor_element_print_func_s32(FILE *os, void *element) {
    fprintf(os, "%d", *(S32*)element);
}

void tensor_fprint(FILE *os, Tensor *tensor) {
    if (tensor->type_hash == Tensor_TypeHash(double)) {
        tensor_fprint_custom(os, tensor, tensor_element_print_func_f64);
    }
    else if (tensor->type_hash == Tensor_TypeHash(uint32_t)) {
        tensor_fprint_custom(os, tensor, tensor_element_print_func_s32);
    }
    else {
        fprintf(stderr, "tensor_print: element type %.*s not supported\n", str8_varg(tensor->type_name));
    }
}

void tensor_print(Tensor *tensor) {
    tensor_fprint(stdout, tensor);
}

static void print_coordinates(FILE *os, U32 *coords, U32 coord_count) {
    fprintf(os, "[");
    for (int i = 0; i < coord_count; ++i) {
        if (i != 0) fprintf(os, ", ");
        fprintf(os, "%d", coords[i]);
    }
    fprintf(os, "]");
}

U64 tensor_element_count(Tensor *t) {
    if (t->ndims == 0) return 0;
    U64 result = 1;
    for (int i = 0; i < t->ndims; ++i) {
        result *= t->shape[i];
    }
    return result;
}

U32 *compute_contiguous_strides(Arena *arena, U32 *shape, U32 ndims) {
    U32 *strides = push_array(arena, U32, ndims);
    for (int i = 0; i < ndims; ++i) {
        strides[i] = 1;
        for (int j = i+1; j < ndims; ++j) {
            strides[i] *= shape[j];
        }
    }
    return strides;
}

// Increments the passed in coords in-place. Returns 0 (false) if coords overflowed.
B32 coord_iter_next(U32 *coords, U32 *shape, U32 ndims) {
    B32 success = 0;
    int i = ndims-1;
    while (i >= 0) {
        coords[i] += 1;
        if (coords[i] >= shape[i]) {
            coords[i] = 0;
            --i;
        }
        else {
            success = 1; // no overflow
            break;
        }
    }
    return success;
}

Tensor *tensor_clone(Arena *arena, Tensor *t) {
    ArenaTemp scratch = scratch_begin(&arena, 1);

    U64 element_count = tensor_element_count(t);

    U64 new_data_size = element_count * t->element_size;
    void *cloned_data = MD_ArenaPush(arena, new_data_size); // pray to the gods this is aligned properly

    if (element_count > 0) {
        U32 *cur_coords = push_array(scratch.arena, U32, t->ndims);
        U64 element_idx = 0;
        do {
            void *src = tensor_get(t, cur_coords, t->ndims);
            void *dest = (char*)cloned_data + (element_idx * t->element_size);
            MemoryCopy(dest, src, t->element_size);
            ++element_idx;
        } while (coord_iter_next(cur_coords, t->shape, t->ndims));
    }
    
    Tensor *result = push_array(arena, Tensor, 1);

    result->data = cloned_data;
    
    result->ndims = t->ndims;
    result->shape = push_array(arena, U32, t->ndims);
    // Copy over shape
    for (int i = 0; i < t->ndims; ++i) {
        result->shape[i] = t->shape[i];
    }
    result->strides = compute_contiguous_strides(arena, t->shape, t->ndims);

    result->element_size = t->element_size;
    result->type_hash = t->type_hash;
    result->type_name = str8_copy(arena, t->type_name);

    scratch_end(scratch);
    return result;
}


B32 tensor_shapes_match(Tensor *x, Tensor *y) {
    if (x->ndims != y->ndims) return 0;
    for (int i = 0; i < x->ndims; ++i) if (x->shape[i] != y->shape[i]) return 0;
    return 1;
}


static inline void tensor_element_add_func_f64(void *dest, void *to_add) {
    *(F64*)dest += *(F64*)to_add;
}

static inline void tensor_element_add_func_s32(void *dest, void *to_add) {
    *(S32*)dest += *(S32*)to_add;
}

Tensor *tensor_add_custom(Arena *arena, Tensor *x, Tensor *y, TensorElementAddFunc *add_func) {
    /* TODO: prefer cloning the tensor with contiguous memory */
    Tensor *to_clone = x;
    Tensor *other = (to_clone == x ? y : x);
    Tensor *result = tensor_clone(arena, to_clone);
    if (tensor_element_count(result) > 0) {
        U32 *cur_coord = push_array(arena, U32, result->ndims);
        do {
            /* TODO: use tensor_get_unchecked for SPEED */
            void *dest = tensor_get(result, cur_coord, result->ndims);
            void *to_add = tensor_get(other, cur_coord, result->ndims);
            add_func(dest, to_add);
        } while(coord_iter_next(cur_coord, result->shape, result->ndims));
    } 
    return result;
}

Tensor *tensor_add(Arena *arena, Tensor *x, Tensor *y) {

    if (!tensor_shapes_match(x, y)) {
        fprintf(stderr, "tensor_add: addition of tensors of differing shapes is not supported\n");
        return 0;
    }

    // TODO: pre-compute primitive type hashes in these comparisons
    if (x->type_hash == Tensor_TypeHash(double)) {
        return tensor_add_custom(arena, x, y, tensor_element_add_func_f64);
    }
    else if (x->type_hash == Tensor_TypeHash(uint32_t)) {
        return tensor_add_custom(arena, x, y, tensor_element_add_func_s32);
    }
    else {
        fprintf(stderr, "tensor_add: addition not supported for element type: %.*s\n", str8_varg(x->type_name));
        fprintf(stderr, "HINT: Call tensor_add_custom with your custom element add function instead.\n");
        return 0;
    }

    return 0;
}


