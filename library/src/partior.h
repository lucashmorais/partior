#ifndef _PARTIOR_LIB
#define _PARTIOR_LIB
extern "C" {
  void hello_partior();
  void cb_initialize_partior();
  void cb_add_task();
  void cb_remove_task();
  void cb_add_dep(unsigned dep_type, unsigned long long address);
  void cb_allocation_hint();
  void cb_finish_adding_task(unsigned long long task_addr);
}
#endif
