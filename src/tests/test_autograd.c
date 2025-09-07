internal
T_TestResultList test_backward(Arena *arena) {
    T_TestResultList result = {0};
    
    ArenaTemp scratch = scratch_begin(&arena, 1);

    // z = a+b*c
    AG_Value *a = push_array(scratch.arena, AG_Value, 1);
    a->value = 10;
    AG_Value *b = push_array(scratch.arena, AG_Value, 1);
    b->value = 20;
    AG_Value *c = push_array(scratch.arena, AG_Value, 1);
    c->value = 30;

    AG_Value *product = ag_mul(scratch.arena, b, c);
    AG_Value *z = ag_add(scratch.arena, a, product);

    ag_backward(z);

    T_TestAssert(arena, &result, a->grad == z->grad);
    T_TestAssert(arena, &result, z->grad == 1);
    T_TestAssert(arena, &result, b->grad == c->value);
    T_TestAssert(arena, &result, c->grad == b->value);

    scratch_end(scratch);
    return result;
}

internal
T_TestResultList test_autograd(Arena *arena) {
    T_TestResultList results = {0};

    T_RunTest(arena, &results, test_backward);

    return results;
}
