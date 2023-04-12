use partior::*;
use rand::Rng;
use std::time::Instant;
use std::num::NonZeroU32;
use permutation_iterator::Permutor;

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

// This produces very high-quality permutations,
// but is very slow for our purposes (0.7 ms per
// iteration)
//
// On the other hand, achieving a similar purpose by
// randomly choosing a number from a range and later
// sorting the whole sequence to efficiently detect
// duplicates would require us to frequently allocate
// and de-allocate vectors.
fn test_hashed_permutation() {
    let max = 100;
    let num_selections = 10;
    let num_iterations = 1000;

    let start = Instant::now();

    for _ in 0..num_iterations {
        let mut remaining_selections = num_selections;

        let permutor = Permutor::new(max);

        for permuted in permutor {
            if remaining_selections == 0 {
                break;
            }
            println!("{}", permuted);
            remaining_selections -= 1;
        }
    }

    println!("Time for selecting {} out of {} a total of {} times: {:?}", num_selections, max, num_iterations, start.elapsed());
}

fn test_sorted_non_duplicate_sampling() {
    let max = 100;
    let num_selections = 10;
    let num_iterations = 1000;

    let start = Instant::now();

    let mut rng = rand::thread_rng();

    for _ in 0..num_iterations {
        let mut vec: Vec<usize> = Vec::with_capacity(num_selections);
        let mut final_vec: Vec<usize> = Vec::with_capacity(num_selections);
        let mut disabled_vec: Vec<bool> = vec![false; num_selections];

        for _ in 0..num_selections {
            let sampled_value = rng.gen_range(0..max);
            vec.push(sampled_value);
        }

        vec.sort();

        for i in 1..num_selections {
            if vec[i] == vec[i - 1] {
                disabled_vec[i] = true;
            }
        }

        for i in 0..num_selections {
            if !disabled_vec[i] {
                final_vec.push(vec[i]);
            }
        }
    }

    println!("Time for selecting {} out of {} a total of {} times: {:?}", num_selections, max, num_iterations, start.elapsed());
}

// This method is substantially faster than the
// sort-based one at least for num_selection not
// higher than 1000.
//
// For num_selection up to 10, it is just around
// 20% slower than duplicate-enabled sampling.
fn test_square_non_duplicate_sampling() {
    let max = 100;
    let num_selections = 10;
    let num_iterations = 1000;

    let start = Instant::now();

    let mut rng = rand::thread_rng();

    for _ in 0..num_iterations {
        let mut vec: Vec<usize> = Vec::with_capacity(num_selections);

        for _ in 0..num_selections {
            let sampled_value = rng.gen_range(0..max);

            if vec.iter().position(|x| *x == sampled_value).is_none() {
                vec.push(sampled_value);
            }
        }
    }

    println!("Time for selecting {} out of {} a total of {} times: {:?}", num_selections, max, num_iterations, start.elapsed());
}

// This is some 10% faster than the method
// that allocates a new vector at the start
// of every iteration.
fn test_square_non_duplicate_sampling_low_allocation() {
    let max = 100;
    let num_selections = 10;
    let num_iterations = 1000;

    let start = Instant::now();

    let mut rng = rand::thread_rng();

    // We avoid allocations by initializing this
    // vector once and clearing it at the end of
    // each iteration.
    let mut vec: Vec<usize> = Vec::with_capacity(num_selections);
    for _ in 0..num_iterations {
        for _ in 0..num_selections {
            let sampled_value = rng.gen_range(0..max);

            if vec.iter().position(|x| *x == sampled_value).is_none() {
                vec.push(sampled_value);
            }
        }

        vec.clear();
    }

    println!("Time for selecting {} out of {} a total of {} times: {:?}", num_selections, max, num_iterations, start.elapsed());
}

fn test_square_non_duplicate_sampling_array() {
    let max = 100;
    let num_selections = 10;
    let num_iterations = 1000;
    const MAX_NUM_SELECTIONS: usize = 10;

    let start = Instant::now();

    let mut rng = rand::thread_rng();

    // We avoid allocations by initializing this
    // vector once and clearing it at the end of
    // each iteration.
    let mut arr: [usize; MAX_NUM_SELECTIONS] = [0; MAX_NUM_SELECTIONS];
    let mut num_elements = 0;
    for _ in 0..num_iterations {
        'outer: for _ in 0..num_selections {
            let sampled_value = rng.gen_range(0..max);

            for i in 0..num_elements {
                if arr[i] == sampled_value {
                    continue 'outer;
                }
            }

            arr[num_elements] = sampled_value;
            num_elements += 1;
        }

        num_elements = 0;
    }

    println!("Time for selecting {} out of {} a total of {} times: {:?}", num_selections, max, num_iterations, start.elapsed());
}

fn test_square_non_duplicate_sampling_array_iter_search() {
    let max = 100;
    let num_selections = 10;
    let num_iterations = 1000;
    const MAX_NUM_SELECTIONS: usize = 10;

    let start = Instant::now();

    let mut rng = rand::thread_rng();

    // We avoid allocations by initializing this
    // vector once and clearing it at the end of
    // each iteration.
    let mut arr: [usize; MAX_NUM_SELECTIONS] = [0; MAX_NUM_SELECTIONS];
    let mut num_elements = 0;
    for _ in 0..num_iterations {
        for _ in 0..num_selections {
            let sampled_value = rng.gen_range(0..max);

            if arr.iter().position(|x| *x == sampled_value).is_none() {
                arr[num_elements] = sampled_value;
                num_elements += 1;
            }
        }

        num_elements = 0;
    }

    println!("Time for selecting {} out of {} a total of {} times: {:?}", num_selections, max, num_iterations, start.elapsed());
}

fn test_sampling() {
    let max = 100;
    let num_selections = 10;
    let num_iterations = 1000;

    let start = Instant::now();

    let mut rng = rand::thread_rng();
    let mut vec: Vec<usize> = Vec::with_capacity(num_selections);

    for _ in 0..num_iterations {
        for _ in 0..num_selections {
            let sampled_value = rng.gen_range(0..max);
            vec.push(sampled_value);
        }

        vec.clear();
    }

    println!("Time for selecting {} out of {} a total of {} times: {:?}", num_selections, max, num_iterations, start.elapsed());
}

fn test_sampling_array() {
    let max = 100;
    let num_selections = 10;
    const MAX_NUM_SELECTIONS: usize = 1000;
    let num_iterations = 1000;

    let start = Instant::now();

    let mut rng = rand::thread_rng();
    let mut arr: [usize; MAX_NUM_SELECTIONS] = [0; MAX_NUM_SELECTIONS];
    let mut num_elements = 0;

    for _ in 0..num_iterations {
        for _ in 0..num_selections {
            let sampled_value = rng.gen_range(0..max);
            arr[num_elements] = sampled_value;
            num_elements += 1;
        }

        num_elements = 0;
    }

    println!("Time for selecting {} out of {} a total of {} times: {:?}", num_selections, max, num_iterations, start.elapsed());
}

fn test_sampling_no_vec() {
    let max = 100;
    let num_selections = 10;
    let num_iterations = 1000;

    let start = Instant::now();

    let mut rng = rand::thread_rng();

    for _ in 0..num_iterations {
        let mut vec: Vec<usize> = Vec::with_capacity(num_selections);

        for _ in 0..num_selections {
            let sampled_value = rng.gen_range(0..max);
        }
    }

    println!("Time for selecting {} out of {} a total of {} times: {:?}", num_selections, max, num_iterations, start.elapsed());
}

fn test_task_dependency_graph_generation_and_retirement() {
    let mut dep_graph = DepGraph::new();
    let mut rng = rand::thread_rng();

    const MAX_OUT_DEPS_PER_TASK: usize = 4;
    const MAX_IN_DEPS_PER_TASK: usize = 4;
    const ADDRESS_SPACE_SIZE: usize = 100;
    const NUM_TASKS: usize = 256;

    let mut read_dep_arr: [usize; MAX_IN_DEPS_PER_TASK] = [0; MAX_IN_DEPS_PER_TASK];
    let mut write_dep_arr: [usize; MAX_OUT_DEPS_PER_TASK] = [0; MAX_OUT_DEPS_PER_TASK];
    let mut num_read_deps = 0;
    let mut num_write_deps = 0;

    let time_gen = Instant::now();
    for i in 0..NUM_TASKS {
        dep_graph.add_task(i);

        let num_out_deps = rng.gen_range(0..=MAX_OUT_DEPS_PER_TASK);
        let num_in_deps = rng.gen_range(0..=MAX_IN_DEPS_PER_TASK);

        for _ in 0..num_out_deps {
            let addr = rng.gen_range(0..ADDRESS_SPACE_SIZE);

            if write_dep_arr.iter().position(|x| *x == addr).is_none() {
                write_dep_arr[num_write_deps] = addr;
                num_write_deps += 1;
            }
        }

        for k in 0..num_write_deps {
            dep_graph.add_task_write_dep(i, write_dep_arr[k]);
        }

        for _ in 0..num_in_deps {
            let addr = rng.gen_range(0..ADDRESS_SPACE_SIZE);

            // We need having two array searches here ensures that
            // no addr is both an IN and an OUT dependence
            if read_dep_arr.iter().position(|x| *x == addr).is_none() && write_dep_arr.iter().position(|x| *x == addr).is_none(){
                read_dep_arr[num_read_deps] = addr;
                num_read_deps += 1;
            }
        }

        for k in 0..num_read_deps {
            dep_graph.add_task_read_dep(i, read_dep_arr[k]);
        }

        dep_graph.finish_adding_task(i);

        num_read_deps = 0;
        num_write_deps = 0;
    }
    let time_gen = time_gen.elapsed();
    println!("Time to generate random graph: {:?}", time_gen);

    // TODO: Remove println from inside time measurement zone
    let time_ret = Instant::now();
    //println!("Number of ready tasks: {}", dep_graph.num_ready_tasks());
    while dep_graph.num_ready_tasks() > 0 {
        //println!("Number of ready tasks: {}", dep_graph.num_ready_tasks());
        let task = dep_graph.pop_ready_task().unwrap();
        dep_graph.retire_task(task);
    }
    let time_ret = time_ret.elapsed();
    println!("Time to retire all tasks: {:?}", time_ret);

    println!("Total time {:?}", time_gen + time_ret);

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
    
    //test_task_dependency_graph_generation();
    test_task_dependency_graph_generation_and_retirement();

    //test_hashed_permutation();
    
    /*
    test_sorted_non_duplicate_sampling();
    test_square_non_duplicate_sampling();
    test_square_non_duplicate_sampling_low_allocation();
    test_square_non_duplicate_sampling_array();
    test_square_non_duplicate_sampling_array_iter_search();
    test_sampling();
    test_sampling_array();
    test_sampling_no_vec();
    */
}
