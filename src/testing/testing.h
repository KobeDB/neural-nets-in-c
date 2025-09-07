#ifndef TESTING_H
#define TESTING_H

typedef struct T_TestResultNode T_TestResultNode;
struct T_TestResultNode {
    String8 file_path;
    int line;
    String8 expression;
    B32 is_failure;

    T_TestResultNode *next;
};

typedef struct T_TestResultList T_TestResultList;
struct T_TestResultList {
    T_TestResultNode *first;
    T_TestResultNode *last;
    int count;
};


#define T_TestAssert(arena, list_ptr, expression) do {\
    String8 file_path = str8_lit(__FILE__); \
    String8 expression_str = str8_lit(#expression); \
    t_push_test_result((arena), (list_ptr), (expression), file_path, __LINE__, expression_str); \
} while(0)

#define T_RunTest(arena, result_list_ptr, test_function) do { \
    printf("Running Test %s...\n", #test_function); \
    T_TestResultList result = (test_function)((arena)); \
    t_concatenate_to_back_of_result_list((result_list_ptr), &result); \
} while (0)


internal void t_push_test_result(Arena *arena, T_TestResultList *list, B32 expression_result, String8 file_path, int line, String8 expression_str);

internal void t_concatenate_to_back_of_result_list(T_TestResultList *to_concatenate_to, T_TestResultList *to_concatenate);

internal void t_print_test_report(T_TestResultList *result_list);



#endif
