use partior::*;
use rand::Rng;
use std::time::Instant;

fn test_task_dependency_graph_generation() {
    let mut dep_graph = DepGraph::new();
    let mut rng = rand::thread_rng();

    const MAX_OUT_DEPS_PER_TASK: usize = 4;
    const MAX_IN_DEPS_PER_TASK: usize = 4;
    const ADDRESS_SPACE_SIZE: usize = 100;
    const NUM_TASKS: usize = 256;

    let start = Instant::now();
    for i in 0..NUM_TASKS {
        dep_graph.add_task(i);

        let num_out_deps = rng.gen_range(0..=MAX_OUT_DEPS_PER_TASK);
        let num_in_deps = rng.gen_range(0..=MAX_IN_DEPS_PER_TASK);

        for _ in 0..num_out_deps {
            let addr = rng.gen_range(0..ADDRESS_SPACE_SIZE);
            dep_graph.add_task_write_dep(i, addr);
        }

        for _ in 0..num_in_deps {
            let addr = rng.gen_range(0..ADDRESS_SPACE_SIZE);
            dep_graph.add_task_read_dep(i, addr);
        }
    }
    println!("Time to generate random graph: {:?}", start.elapsed());

    dep_graph.visualize_graph(Some(format!("task_dependency_graph")));
}

fn main() {
    //gen_sample_graph_image();
    //test_histogram_01();
    //test_metaheuristics_01();
    //test_metaheuristics_02();
    
    //test_metaheuristics_03(10);
    //test_tree_transform();
    //test_n_tree_transform();
    //test_multi_level_clustering(false);
    //test_local_surprise_search();
    //test_local_permanence_search();
    //test_dinitz_max_flow();
    //export_clustering_problem();
    //import_and_visualize_graph();

    //test_gusfield_gomory_hu_solver();
    //hello_partior();
    //test_max_flow_clustering();
    
    test_task_dependency_graph_generation();
}
