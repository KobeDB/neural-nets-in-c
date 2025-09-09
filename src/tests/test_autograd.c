internal
T_TestResultList test_backward(Arena *arena) {
    T_TestResultList result = {0};
    
    ArenaTemp scratch = scratch_begin(&arena, 1);

    {
        // z = a+b*c
        AG_Value *a = ag_source(scratch.arena, 10);
        AG_Value *b = ag_source(scratch.arena, 20);
        AG_Value *c = ag_source(scratch.arena, 30);

        AG_Value *product = ag_mul(scratch.arena, b, c);
        AG_Value *z = ag_add(scratch.arena, a, product);

        ag_backward(z);

        T_TestAssert(arena, &result, a->grad == z->grad);
        T_TestAssert(arena, &result, z->grad == 1);
        T_TestAssert(arena, &result, b->grad == c->value);
        T_TestAssert(arena, &result, c->grad == b->value);
    }
    {
        // z = a+a*a
        AG_Value *a = ag_source(scratch.arena, 10);

        AG_Value *z = ag_add(scratch.arena, a, ag_mul(scratch.arena, a, a));

        ag_backward(z);

        T_TestAssert(arena, &result, z->grad == 1);
        T_TestAssert(arena, &result, a->grad == 1+2*a->value);
    }
    {
        // z = a - b
        AG_Value *a = ag_source(scratch.arena, 10);
        AG_Value *b = ag_source(scratch.arena, 20);

        AG_Value *z = ag_sub(scratch.arena, a, b);

        ag_backward(z);

        T_TestAssert(arena, &result, z->grad == 1);
        T_TestAssert(arena, &result, a->grad == 1);
        T_TestAssert(arena, &result, b->grad == -1);
        T_TestAssert(arena, &result, z->value == -10);
    }
    {
        // z = relu(a)
        AG_Value *a = ag_source(scratch.arena, 10);

        AG_Value *z = ag_relu(scratch.arena, a);
        
        ag_backward(z);

        T_TestAssert(arena, &result, z->grad == 1);
        T_TestAssert(arena, &result, a->grad == 1);

        // zero grad
        a->grad = 0;
        z->grad = 0;

        a->value = -10;
        ag_backward(z);
        
        T_TestAssert(arena, &result, z->grad == 1);
        T_TestAssert(arena, &result, a->grad == 0);
    }
    {
        // z = pow(a, k)
        AG_Value *a = ag_source(scratch.arena, 10);
        AG_Value *z = ag_pow(scratch.arena, a, 3);
        ag_backward(z);
        T_TestAssert(arena, &result, a->grad == 3*a->value*a->value);
        T_TestAssert(arena, &result, z->value == 1000);
    }

    scratch_end(scratch);
    return result;
}

internal
T_TestResultList test_autograd(Arena *arena) {
    T_TestResultList results = {0};

    T_RunTest(arena, &results, test_backward);

    return results;
}
