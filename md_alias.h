#ifndef MD_ALIAS_H
#define MD_ALIAS_H

//
// Primitive Types
//

typedef MD_i8  S8;
typedef MD_i16 S16;
typedef MD_i32 S32;
typedef MD_i64 S64;

typedef MD_u8  U8;
typedef MD_u16 U16;
typedef MD_u32 U32;
typedef MD_u64 U64;

typedef MD_f32 F32;
typedef MD_f64 F64;

typedef MD_b8  B8;
typedef MD_b16 B16;
typedef MD_b32 B32;
typedef MD_b64 B64;

//
// Strings
// 
typedef MD_String8      String8;
typedef MD_String8List  String8List;
typedef MD_StringJoin   StringJoin;
typedef MD_String8Node  String8Node;

#define str8_split        MD_S8Split 
#define str8_list_join    MD_S8ListJoin

#define str8_varg(s)      MD_S8VArg(s)   
#define str8_lit(s)       MD_S8Lit(s)

#define str8_skip_whitespace MD_S8SkipWhitespace
#define str8_prefix MD_S8Prefix
#define str8_skip MD_S8Skip
#define str8_match MD_S8Match
#define str8_find_substring MD_S8FindSubstring
#define str8_substring MD_S8Substring
#define str8_skip MD_S8Skip
#define str8_copy MD_S8Copy

//
// Characters
//

#define char_is_digit MD_CharIsDigit
#define c_style_int_from_string MD_CStyleIntFromString

//
// Arena
//
typedef MD_Arena        Arena;
typedef MD_ArenaTemp    ArenaTemp;

#define scratch_begin                 MD_GetScratch
#define scratch_end(scratch)    MD_ReleaseScratch(scratch)

#define arena_alloc                 MD_ArenaAlloc
#define push_array(a,T,c)           MD_PushArrayZero(a,T,c)
#define push_array_no_zero(a,T,c)   MD_PushArray(a,T,c)

#define temp_begin MD_ArenaBeginTemp
#define temp_end MD_ArenaEndTemp

//
// Linked List Macros
//

#define SLLQueuePush(f,l,n) MD_QueuePush(f,l,n)
#define SLLQueuePop(f,l)    MD_QueuePop(f,l)
#define SLLStackPush(f,n)   MD_StackPush(f,n)

//
// Memory Ops
//

#define MemoryCopy(d,s,z) MD_MemoryCopy(d,s,z)

#define ArrayCopy(d,s,c) do {\
    for (int _aci = 0; _aci < (c); ++_aci ) (d)[_aci] = (s)[_aci];\
} while (0)

#define ArrayCount(a) MD_ArrayCount(a)

//
// Assert
//

#define Assert(c) MD_Assert(c)

#endif