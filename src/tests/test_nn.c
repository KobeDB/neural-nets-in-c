T_TestResultList test_neuron(Arena *arena) {
    T_TestResultList test_results = {0};

    ArenaTemp scratch = scratch_begin(&arena, 1);

    B32 has_relu = 1;
    F64 weights[] = {2, -1, 3};
    int input_dim = ArrayCount(weights);
    F64 bias = -5;
    NN_Neuron n = nn_make_neuron_from_weights(scratch.arena, weights, ArrayCount(weights), bias, has_relu);

    T_TestAssert(arena, &test_results, nn_neuron_get_params(scratch.arena, &n).count == input_dim+1);

    F64 x_raw[] = {10,20,30};
    AG_ValueArray x = ag_value_array_from_raw(scratch.arena, x_raw, ArrayCount(x_raw));
    
    AG_Value *n_result = nn_neuron_apply(scratch.arena, &n, x);

    F64 weighted_sum = bias;
    for (int i = 0; i < input_dim; ++i) weighted_sum += weights[i]*x_raw[i];
    T_TestAssert(arena, &test_results, n_result->value == weighted_sum);

    scratch_end(scratch);
    return test_results;
}

T_TestResultList test_layer(Arena *arena) {
    T_TestResultList test_results = {0};

    ArenaTemp scratch = scratch_begin(&arena, 1);

    F64 x_raw[] = {1,2,3};
    F64 w0_raw[] = {4,5,6};
    F64 w1_raw[] = {7,8,9};
    F64 b0 = 1, b1 = 2;
    int input_dim = ArrayCount(x_raw);
    int output_dim = 2;

    AG_ValueArray x = ag_value_array_from_raw(scratch.arena, x_raw, ArrayCount(x_raw));

    NN_Layer layer = nn_make_layer_with_random_init(scratch.arena, input_dim, output_dim, 0);
    layer.neurons[0] = nn_make_neuron_from_weights(scratch.arena, w0_raw, ArrayCount(w0_raw), b0, 0);
    layer.neurons[1] = nn_make_neuron_from_weights(scratch.arena, w1_raw, ArrayCount(w1_raw), b1, 0);

    T_TestAssert(arena, &test_results, nn_layer_get_params(scratch.arena, &layer).count == ArrayCount(w0_raw)+ArrayCount(w1_raw)+1+1);

    AG_ValueArray lresult = nn_layer_apply(scratch.arena, scratch.arena, &layer, x);

    F64 expected[2] = {b0, b1};
    for (int i = 0; i < input_dim; ++i) {
        expected[0] += x_raw[i] * w0_raw[i];
        expected[1] += x_raw[i] * w1_raw[i];
    } 
    T_TestAssert(arena, &test_results, lresult.values[0]->value == expected[0]);
    T_TestAssert(arena, &test_results, lresult.values[1]->value == expected[1]);


    scratch_end(scratch);
    return test_results;
}

T_TestResultList test_nn(Arena *arena) {
    T_TestResultList test_results = {0};

    ArenaTemp scratch = scratch_begin(&arena, 1);

    T_RunTest(arena, &test_results, test_neuron);
    T_RunTest(arena, &test_results, test_layer);

    scratch_end(scratch);
    return test_results;
}
