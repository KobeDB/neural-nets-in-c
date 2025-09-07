T_TestResultList test_nn(Arena *arena) {
    T_TestResultList test_results = {0};

    ArenaTemp scratch = scratch_begin(&arena, 1);

    {
        B32 has_relu = 1;
        F64 n_weights[] = {2, -1, 3};
        int input_dim = ArrayCount(n_weights);
        NN_Neuron n = nn_make_neuron(scratch.arena, input_dim, has_relu, 42);
        n.weights = ag_value_array_from_raw(scratch.arena, n_weights, ArrayCount(n_weights));
        n.bias = ag_source(scratch.arena, -5);

        F64 x_raw[] = {10,20,30};
        AG_ValueArray x = ag_value_array_from_raw(scratch.arena, x_raw, ArrayCount(x_raw));
        
        AG_Value *n_result = nn_neuron_apply(scratch.arena, &n, x);

        F64 weighted_sum = n.bias->value;
        for (int i = 0; i < n.weights.count; ++i) weighted_sum += n_weights[i]*x_raw[i];
        T_TestAssert(arena, &test_results, n_result->value == weighted_sum);
    }

    scratch_end(scratch);
    return test_results;
}
