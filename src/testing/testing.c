internal
void t_push_test_result(Arena *arena, T_TestResultList *list, B32 expression_result, String8 file_path, int line, String8 expression_str) {
    T_TestResultNode *node = push_array(arena, T_TestResultNode, 1);
    node->file_path = file_path;
    node->line = line;
    node->expression = expression_str;
    node->is_failure = !expression_result;
    SLLQueuePush(list->first, list->last, node);
    list->count += 1;
}

internal
void t_concatenate_to_back_of_result_list(T_TestResultList *to_concatenate_to, T_TestResultList *to_concatenate) {
    if (!to_concatenate_to->last) {
        *to_concatenate_to = *to_concatenate;
    }
    else {
        to_concatenate_to->last->next = to_concatenate->first;
        to_concatenate_to->last = to_concatenate->last;
        to_concatenate_to->count += to_concatenate->count;
    }
}

internal
void t_print_test_report(T_TestResultList *result_list) {
    int total = 0;
    int failures = 0;
    for (T_TestResultNode *result = result_list->first; result; result=result->next) {
        if (result->is_failure) { 
            printf("[FAIL] %.*s:%d | %.*s\n", str8_varg(result->file_path), result->line, str8_varg(result->expression));
            failures += 1;
        } else { 
            printf("[PASS] %.*s:%d | %.*s\n", str8_varg(result->file_path), result->line, str8_varg(result->expression)); 
        }
        total += 1;
    }

    printf("SUMMARY | failed: %d | total: %d | \n", failures, total);
}
