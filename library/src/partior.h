#ifndef _PARTIOR_LIB
#define _PARTIOR_LIB
extern "C" {
  void hello_partior();
  void cb_initialize_partior();
  void cb_add_task();
  void cb_remove_task();
  void cb_add_dep();
  void cb_allocation_hint();
}
#endif
