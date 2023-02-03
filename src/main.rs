use petgraph::graph::*;
use petgraph::graphmap::GraphMap;
use petgraph::visit::IntoNodeReferences;
use petgraph::visit::NodeRef;
use petgraph::algo::{dijkstra, min_spanning_tree};
use petgraph::data::FromElements;
use petgraph::dot::{Dot, Config};
use std::fs::File;
use std::io::prelude::*;
use std::fmt;
use std::process::Command;
use rand::Rng;
use std::cmp::min;
use statrs::function::factorial::binomial;
use std::collections::HashMap;
use csv::Writer;
use std::time::{Duration, Instant};

use metaheuristics_nature::{Rga, Fa, Pso, De, Tlbo, Solver};
use metaheuristics_nature::tests::TestObj;

const MAX_NUMBER_OF_PARTITIONS: usize = 8;
const MAX_NUMBER_OF_NODES: usize = 256;

/*
use num_integer::binomial;
use num::FromPrimitive;
use num::bigint::BigInt;
use num::cast::ToPrimitive;
*/

pub trait HasPartition {
    fn partition_id(&self) -> usize;
}

#[derive(Clone,Copy,Eq,PartialEq,PartialOrd,Hash,Ord)]
struct NodeInfo {
    numerical_id: usize,
    partition_id: usize
}

impl fmt::Display for NodeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.numerical_id)
    }
}

impl fmt::Debug for NodeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        //write!(f, "{}", self.numerical_id)
        write!(f, "{}, {}", self.numerical_id, self.partition_id)
    }
}

impl HasPartition for NodeInfo {
  fn partition_id(&self) -> usize {
    self.partition_id
  }
}

#[allow(dead_code)]
const COLORS: &'static [&'static str] = &["#d9ed92", "#b5e48c", "#99d98c", "#76c893", "#52b69a", "#34a0a4", "#168aad", "#1a759f", "#1e6091", "#184e77", "#797d62", "#9b9b7a", "#baa587", "#d9ae94", "#f1dca7", "#ffcb69", "#e8ac65", "#d08c60", "#b58463", "#997b66", "#250902", "#38040e", "#640d14", "#800e13", "#ad2831"];

// Here I use the ampersend to prevent this function
// from taking ownership of the string that is passed
// to it. This is useful because if I took the value
// itself rather than the reference and wanted the
// caller of this function to continue being able to
// use the string after the call, this function would
// need to return the String at the end of this body.
// Otherwise, as it finished, the String value that
// it would still hold the ownership of would go out
// of scope and thus be dropped from memory.
fn write_to_file(s: &String, output: String) {
    const TEMP_FILE_NAME: &str = "temp.dot";
    let mut f = File::create(TEMP_FILE_NAME).unwrap();
    //println!("{}", s);
    f.write(s.as_bytes()).unwrap();

    gen_graph_image(TEMP_FILE_NAME, output);
}

// Create an undirected graph with `i32` nodes and edges with `()` associated data.
fn _gen_sample_graph() -> UnGraph<i32, ()> {
    let g = UnGraph::<i32, ()>::from_edges(&[
        (1, 2), (2, 3), (3, 4),
        (1, 4)]);

    // Find the shortest path from `1` to `4` using `1` as the cost for every edge.
    let node_map = dijkstra(&g, 1.into(), Some(4.into()), |_| 1);
    assert_eq!(&1i32, node_map.get(&NodeIndex::new(4)).unwrap());

    // Get the minimum spanning tree of the graph as a new graph, and check that
    // one edge was trimmed.
    let mst = UnGraph::<_, _>::from_elements(min_spanning_tree(&g));
    assert_eq!(g.raw_edges().len() - 1, mst.raw_edges().len());

    return mst;
}

pub trait CanCountInternalLinks {
    fn internal_edge_count(&self, pid_array: Option<&[usize]>) -> usize;
    fn calculate_max_internal_links(&self, pid_array: Option<&[usize]>) -> usize;
}

impl CanCountInternalLinks for Graph<NodeInfo, usize, petgraph::Directed, usize> {
    fn internal_edge_count(&self, pid_array: Option<&[usize]>) -> usize {
        let mut num_internal_links : usize = 0;

        if pid_array.is_none() {
            for v in self.node_references() {
                let pid = v.weight().partition_id();
                //println!("[internal_edge_count]: Visiting node {:?}, belonging to partition {}", v, pid);

                for n in self.neighbors(v.id()) {
                    //println!("[internal_edge_count]: Visiting neighbor {:?}", n);
                    let neighbor_weight = self.node_weight(n).unwrap();
                    let npid = neighbor_weight.partition_id;
                    
                    if  npid == pid {
                        num_internal_links += 1;
                    }
                }
            }
        } else {
            for v in self.node_references() {
                let pid_array = pid_array.unwrap();

                let pid = pid_array[v.weight().numerical_id];
                //println!("[internal_edge_count]: Visiting node {:?}, belonging to partition {}", v, pid);

                for n in self.neighbors(v.id()) {
                    //println!("[internal_edge_count]: Visiting neighbor {:?}", n);
                    let neighbor_weight = self.node_weight(n).unwrap();
                    let npid = pid_array[neighbor_weight.numerical_id];
                    
                    if  npid == pid {
                        num_internal_links += 1;
                    }
                }
            }
        }

        //println!("[internal_edge_count]: num_internal_links = {}", num_internal_links);
        num_internal_links
    }

    fn calculate_max_internal_links(&self, pid_array: Option<&[usize]>) -> usize {
        let mut items_per_partition = HashMap::new();
        let mut max_internal_links = 0;

        if pid_array.is_none() {
            for v in self.node_references() {
                let pid = v.weight().partition_id();

                let hash_count = items_per_partition.entry(pid).or_insert(0);
                *hash_count += 1;
            }
        } else {
            let pid_array = pid_array.unwrap();

            for v in self.node_references() {
                let pid = pid_array[v.weight().numerical_id];

                let hash_count = items_per_partition.entry(pid).or_insert(0);
                *hash_count += 1;
            }
        }

        for n in items_per_partition.values() {
            max_internal_links += n * (n - 1) / 2;
        }

        //println!("[calculate_max_internal_links]: Partition size HashMap: {:?}", items_per_partition);
        //println!("[calculate_max_internal_links]: max_internal_links: {}", max_internal_links);

        max_internal_links
    }
}

#[allow(dead_code)]
fn half_factorial(a: f64, b: f64) -> f64 {
    let rounded_a = a.round() as usize;
    let rounded_b = b.round() as usize;
    let mut partial = 1.0;

    for i in rounded_a..=rounded_b {
        partial *= i as f64;
    }

    partial
}

fn new_binomial(a_raw: u64, b_raw: u64) -> f64 {
    // a: large
    // a: small
    let mut partial = 0.0;

    for i in 0..(a_raw - b_raw) {
        partial += ((b_raw + 1 + i) as f64).log10();
        partial -= ((1 + i) as f64).log10();
    }

    partial
}

// Implemented according to the definition found in https://doi.org/10.1371/journal.pone.0024195 .
// Better clusterings have a higher surprise value.
#[allow(dead_code)]
fn original_calculate_surprise(g: &Graph<NodeInfo, usize, petgraph::Directed, usize>, pid_array: Option<&[usize]>) -> f64 {
    let num_nodes = g.node_count();

    let num_links : u64 = g.edge_count() as u64;
    let num_max_links : u64 = (num_nodes * (num_nodes - 1) / 2) as u64;

    let num_internal_links : u64 = g.internal_edge_count(pid_array) as u64;
    let num_max_internal_links = g.calculate_max_internal_links(pid_array) as u64;

    let top = min(num_links, num_max_internal_links);
    let mut surprise: f64 = 0.0;

    for j in num_internal_links..=top {
        surprise -= (binomial(num_max_internal_links, j)) * (binomial(num_max_links - num_max_internal_links, num_links - j));
    }
    surprise /= binomial(num_max_links, num_links);

    //println!("Graph surprise: {}", surprise);

    surprise
}

// Implemented according to the definition found in https://doi.org/10.1371/journal.pone.0024195 .
// Better clusterings have a lower surprise value.
fn calculate_surprise(g: &Graph<NodeInfo, usize, petgraph::Directed, usize>, pid_array: Option<&[usize]>) -> f64 {
    let num_nodes = g.node_count();

    let num_links : u64 = g.edge_count() as u64;
    let num_max_links : u64 = (num_nodes * (num_nodes - 1) / 2) as u64;

    let num_internal_links : u64 = g.internal_edge_count(pid_array) as u64;
    let num_max_internal_links = g.calculate_max_internal_links(pid_array) as u64;

    let mut surprise: f64 = 0.0;

    let top = min(num_links, num_max_internal_links);

    let j = (num_internal_links + top) / 2;
    surprise += new_binomial(num_max_internal_links, j) + new_binomial(num_max_links - num_max_internal_links, num_links - j);

    //surprise -= new_binomial(num_max_links, num_links);

    //println!("Graph surprise: {}", surprise);

    surprise
}

fn node_par_region(i: usize, min_parallelism: usize) -> usize {
    i / min_parallelism
}

fn first_after_par_region(node_pos: usize, min_parallelism: usize) -> usize {
    (node_par_region(node_pos, min_parallelism) + 1) * min_parallelism
}

fn max_edges_for_given_parallelism(num_nodes: usize, min_parallelism: usize) -> usize {
    let num_complete_groups = num_nodes / min_parallelism;
    let nodes_per_complete_group = min_parallelism;
    let nodes_in_incomplete_group = num_nodes % min_parallelism;

    num_complete_groups * (nodes_per_complete_group * (nodes_per_complete_group / 2)) + (nodes_in_incomplete_group * (nodes_in_incomplete_group) / 2)
}

fn gen_random_digraph(acyclic: bool, max_num_nodes: usize, exact_num_nodes: Option<usize>, max_num_edges: usize, exact_num_edges: Option<usize>, min_parallelism: Option<usize>) -> petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize> {
    let mut g = GraphMap::<NodeInfo, usize, petgraph::Directed>::new();
    let mut rng = rand::thread_rng();

    let num_edges: usize;

    let num_nodes = exact_num_nodes.unwrap_or(rng.gen_range(1..=max_num_nodes));
    let min_parallelism = min_parallelism.unwrap_or(1);

    if exact_num_edges.is_none() {
        if max_num_edges != 0 {
            num_edges = rng.gen_range(0..=max_num_edges);
        } else {
            num_edges = rng.gen_range(0..=(num_nodes * (num_nodes - 1) / 2));
        }
    } else {
        num_edges = exact_num_edges.unwrap();
    }

    println!("Number of nodes: {}", num_nodes);
    println!("Number of edges: {}", num_edges);

    for i in 0..num_nodes {
        g.add_node(NodeInfo{numerical_id: i, partition_id: 0});
    }

    assert!(num_edges <= max_edges_for_given_parallelism(num_nodes, min_parallelism));

    while g.edge_count() < num_edges {
        let i = g.edge_count();
        // These do not need to be
        // mutable beceause we are
        // only assinging any value
        // to them once.
        let a: usize;
        let b: usize;

        if !acyclic {
            a = rng.gen_range(0..num_nodes);
            b = rng.gen_range(0..num_nodes);
        } else {
            a = rng.gen_range(0..num_nodes - 1);
            let first_option = first_after_par_region(a, min_parallelism);
            if num_nodes - first_option <= 0 {
                continue;
            }
            b = rng.gen_range(first_option..num_nodes);
        }

        g.add_edge(NodeInfo {numerical_id: a, partition_id: 0}, NodeInfo {numerical_id: b, partition_id: 0}, i);
    }
    
    let g : petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize> = g.into_graph();

    // The following is equivalent to 'return g;'
    g
}

fn gen_scatter_evolution(csv_path_with_suffix: &'static str, image_output_without_suffix: &'static str) {
    let _dot_proc_output = Command::new("src/metrics_as_function_of_iterations.py").arg(csv_path_with_suffix).arg(image_output_without_suffix).output().unwrap();
}

fn gen_histogram(csv_path_with_suffix: &'static str, image_output_without_suffix: &'static str) {
    let _dot_proc_output = Command::new("src/histogram.py").arg(csv_path_with_suffix).arg(image_output_without_suffix).output().unwrap();
}

fn gen_graph_image(file_name: &'static str, output_name: String) {
    let _dot_proc_output = Command::new("dot").arg("-Tpng").arg(file_name).arg("-o").arg(format!("{output_name}.png")).output().unwrap();
}

fn set_random_partitions(g: &mut petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, max_partitions: usize) {
    let mut rng = rand::thread_rng();
    let num_nodes = g.node_count();

    let max_pid;
    if max_partitions > 0 {
        max_pid = max_partitions;
    } else {
        max_pid = num_nodes;
    }

    for w in g.node_weights_mut() {
        //println!("(nid, pid) = ({}, {})", w.numerical_id, w.partition_id);

        let new_pid = rng.gen_range(0..max_pid);
        w.partition_id = new_pid;
    }

    /*
    for w in g.node_weights() {
        println!("(nid, pid) = ({}, {})", w.numerical_id, w.partition_id);
    }
    */
}

fn randomize_pid_array (pid_array: &mut [usize], num_nodes: usize, max_partitions: usize) {
    let max_pid;
    if max_partitions > 0 {
        max_pid = max_partitions;
    } else {
        max_pid = num_nodes;
    }

    let mut rng = rand::thread_rng();
    for i in 0..num_nodes {
        pid_array[i] = rng.gen_range(1..=max_pid);
    }
}

#[allow(dead_code)]
fn get_number_of_partitions(g: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>) -> usize {
    let mut items_per_partition = HashMap::new();

    for v in g.node_references() {
        let pid = v.weight().partition_id();
        let _ = items_per_partition.entry(pid).or_insert(0);
    }

    items_per_partition.len()
}

fn visualize_graph(g: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, pid_array: Option<&[usize]>, output_name: Option<String>) {
    let null_out = |_, _| "".to_string();
    let mut g = g.clone();

    let out_name_unwrapped;
    if output_name.is_some() {
        out_name_unwrapped = output_name.unwrap();
    } else {
        out_name_unwrapped = "output".to_string();
    }
        
    //let num_partitions = get_number_of_partitions(g);
    
    if pid_array.is_some() {
        let pid_array = pid_array.unwrap();

        for w in g.node_weights_mut() {
            let new_pid = pid_array[w.numerical_id];
            w.partition_id = new_pid;
        }
    }

    fn node_attr_generator<P: petgraph::visit::NodeRef>(_: &Graph<NodeInfo, usize, petgraph::Directed, usize>, node_ref: P) -> String where <P as petgraph::visit::NodeRef>::Weight: fmt::Debug + HasPartition {
        //let new_node_ref: NodeIndex;
        //new_node_ref = node_ref.into();

        let w = node_ref.weight();
        //let c = COLORS[w.partition_id()];
        let c = get_equally_hue_spaced_hsv_string(w.partition_id(), MAX_NUMBER_OF_PARTITIONS);
        format!("style=filled, color=\"{}\", fillcolor=\"{}\"", c, c).to_string()
    }

    let dot_dump = format!("{:?}", Dot::with_attr_getters(&g, &[Config::EdgeNoLabel], &null_out, &node_attr_generator));
    
    let _ = write_to_file(&dot_dump, out_name_unwrapped);
    //calculate_surprise(&g);
}

fn set_random_partitions_and_visualize_graph(graph: &mut petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, max_partitions: usize) -> &Graph<NodeInfo, usize, petgraph::Directed, usize>{
    set_random_partitions(graph, max_partitions);
    visualize_graph(&graph, None, None);
    graph
}

fn get_equally_hue_spaced_hsv_string(index: usize, num_items: usize) -> String {
    let h: f64 = (index as f64) / (num_items as f64);
    let res = format!("{} 0.600 1.00", h);
    //println!("{}", res);
    res
}

fn evaluate_multiple_random_clusterings(original_graph: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, max_partitions: usize, num_iterations: usize, gen_image: bool) {
    const CSV_PATH_WITH_SUFFIX : &str = "surprise.csv";
    const HIST_PATH_WITHOUT_SUFFIX : &str = "surprise_hist";
    
    //let mut g = original_graph.clone();
    let g = original_graph.clone();
    let mut wtr = Writer::from_path(CSV_PATH_WITH_SUFFIX).unwrap();

    // Header
    wtr.write_record(&["surprise"]).unwrap();

    let mut best_surprise_so_far: f64 = -1.1;

    let mut pid_array: [usize; MAX_NUMBER_OF_NODES] = [0; MAX_NUMBER_OF_NODES];
    let num_nodes = g.node_count();

    for i in 0..num_iterations {
        //set_random_partitions(&mut g, max_partitions);

        randomize_pid_array(&mut pid_array, num_nodes, max_partitions);

        let s = calculate_surprise(&g, Some(&pid_array));
        if s > best_surprise_so_far {
            if gen_image {
                visualize_graph(&g, Some(&pid_array), None);
            }

            best_surprise_so_far = s;
            println!("[evaluate_multiple_random_clusterings]: (iteration, best_surprise_so_far): ({}, {})", i, s);
        }

        wtr.write_record(&[s.to_string()]).unwrap();
    }

    wtr.flush().unwrap();
    gen_histogram(CSV_PATH_WITH_SUFFIX, HIST_PATH_WITHOUT_SUFFIX);
    println!("[evaluate_multiple_random_clusterings]: Finished execution.");
}

#[allow(dead_code)]
fn gen_sample_graph_image() {
    //let g = gen_sample_graph();
    let mut g = gen_random_digraph(true, 16, Some(16), 0, None, None);
    let new_graph = set_random_partitions_and_visualize_graph(&mut g, MAX_NUMBER_OF_PARTITIONS);
    visualize_graph(&new_graph, None, None);
}

#[allow(dead_code)]
fn test_histogram_01() {
    let g = gen_random_digraph(true, 16, Some(16), 0, None, None);
    evaluate_multiple_random_clusterings(&g, MAX_NUMBER_OF_PARTITIONS, 100000000, true);
}

#[allow(dead_code)]
fn test_metaheuristics_01() {
    let mut report = Vec::with_capacity(20);


// Build and run the solver
    let s = Solver::build(Rga::default(), TestObj::new())
        .task(|ctx| ctx.gen == 20)
        .callback(|ctx| report.push(ctx.best_f))
        .solve()
        .unwrap();

    // Get the result from objective function
    let ans = s.result();
    // Get the optimized XY value of your function
    let xs = s.best_parameters();
    let y = s.best_fitness();
    // Get the history reports
    let y2 = report[2];

    println!("(ans, xs, y, y2) = ({:?}, {:?}, {:?}, {:?})", ans, xs, y, y2);
}

use metaheuristics_nature::Bounded;
use metaheuristics_nature::ObjFactory;

struct MyFunc<'a> {
    graph: &'a petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>
}

impl Bounded for MyFunc<'_> {
    fn bound(&self) -> &[[f64; 2]] {
        &[[0., MAX_NUMBER_OF_PARTITIONS as f64]; MAX_NUMBER_OF_NODES]
    }
}

fn round_float_array(float_arr: &[f64]) -> [usize; MAX_NUMBER_OF_NODES] {
    let mut int_arr: [usize; MAX_NUMBER_OF_NODES] = [0; MAX_NUMBER_OF_NODES];

    for i in 0..float_arr.len() {
        int_arr[i] = float_arr[i].round() as usize;
    }

    int_arr
}

fn floor_float_array(float_arr: &[f64]) -> [usize; MAX_NUMBER_OF_NODES] {
    let mut int_arr: [usize; MAX_NUMBER_OF_NODES] = [0; MAX_NUMBER_OF_NODES];

    for i in 0..float_arr.len() {
        int_arr[i] = float_arr[i] as usize;
    }

    int_arr
}

impl ObjFactory for MyFunc<'_> {
    type Product = [usize; MAX_NUMBER_OF_NODES];
    type Eval = f64;

    fn produce(&self, xs: &[f64]) -> Self::Product {
        //round_float_array(xs)
        floor_float_array(xs)
    }

    fn evaluate(&self, x: [usize; MAX_NUMBER_OF_NODES]) -> Self::Eval {
        //400. + (-original_calculate_surprise(&self.graph, Some(&x))).log10()
        //1000.0 + calculate_surprise(&self.graph, Some(&x))
        calculate_surprise(&self.graph, Some(&x))
    }
}

#[allow(dead_code)]
// https://docs.rs/metaheuristics-nature/8.0.4/metaheuristics_nature/trait.ObjFunc.html
fn test_metaheuristics_02() {
    let mut report = Vec::with_capacity(20);
    let g = gen_random_digraph(true, 16, Some(32), 64, Some(96), None);

// Build and run the solver
    //let s = Solver::build(Rga::default(), MyFunc{graph: &g})
    //let s = Solver::build(Pso::default(), MyFunc{graph: &g})
    //let s = Solver::build(Tlbo::default(), MyFunc{graph: &g})

    //let s = Solver::build(De::default(), MyFunc{graph: &g})
    let s = Solver::build(Fa::default(), MyFunc{graph: &g})
        .task(|ctx| ctx.gen == 30)
        //.pop_num(20)
        .pop_num(30)
        //.pop_num(30)
        .callback(|ctx| report.push(ctx.best_f))
        .solve()
        .unwrap();

    // Get the result from objective function
    let ans = s.result();
    // Get the optimized XY value of your function
    let xs = s.best_parameters();
    let y = s.best_fitness();
    // Get the history reports
    let y2 = report[2];

    println!("best fitness: {:?}", y);
    println!("");
    println!("answer: {:?}", ans);
    println!("fitness_history:");
    for h in report {
        println!("{}", h);
    }

    visualize_graph(&g, Some(&ans), None);
}

fn write_report_and_clear (name: &str, rep: &mut Vec<f64>, f: &mut File) {
    //f.write(format!("#ALG# {}\n", name).as_bytes()).unwrap();
    let num_elements = rep.len();

    //for h in &mut *rep {
    for i in 0..num_elements {
        let h = rep[i];
        f.write(format!("{name},{i},{h}\n").as_bytes()).unwrap();
    }
    //f.write(format!("\n").as_bytes()).unwrap();

    rep.clear();
}

// TODO: Finish this
// Ref: https://doc.rust-lang.org/reference/macros-by-example.html
macro_rules! run_algo {
    ($t:expr, $report:ident, $g:ident) => {
        let _s = Solver::build($t, MyFunc{graph: &$g})
            .task(|ctx| ctx.gen == NUM_ITER)
            .pop_num(POP_SIZE)
            .callback(|ctx| $report.push(ctx.best_f))
            .solve()
            .unwrap();
    };
}

#[allow(dead_code)]
#[derive(Debug)]
struct ExecutionInfo {
    exec_time: usize,
    total_cpu_time: usize,
    speedup: f64
}

fn find_min_and_index(v: std::slice::Iter<usize>) -> (usize, usize) {
    let mut min = usize::MAX;
    let mut min_index = 0;

    for x in v.enumerate() {
        if *x.1 < min {
            min = *x.1;
            min_index = x.0;
        }
    }

    (min, min_index)    
}

fn find_earlier_task_start_time_and_index(v: &[usize], minimum_task_start_time: &[usize], pid_array: &[usize], target_pid: usize, min_time: usize) -> Option<(usize, usize)> {
    let mut min = usize::MAX;
    let mut min_index = 0;

    let mut res = None;

    for x in v.iter().enumerate() {
        if pid_array[x.0] == target_pid && *x.1 < min && *x.1 >= min_time {
            min = *x.1;
            min_index = x.0;

            res = Some((min, min_index));
        }
    }

    res
}

fn old_evaluate_execution_time_and_speedup(graph: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, pid_array: &[usize]) -> ExecutionInfo {
    let mut next_idle_cycle = [0; MAX_NUMBER_OF_PARTITIONS];
    let mut minimum_task_start_time = [0; MAX_NUMBER_OF_NODES];

    const TASK_SIZE: usize = 2;
    const COMM_PENALTY: usize = 1;

    let mut remaining_nodes = graph.node_count();

    while remaining_nodes > 0 {
        let start_time;
        let core_id;
        (start_time, core_id) = find_min_and_index(next_idle_cycle.iter());

        //(task_start_time, task_id)
    }

    for v in graph.node_references() {
        let id = v.weight().numerical_id;
        let pid = pid_array[id];

        //println!("{:?}, {}", v.weight().numerical_id, pid);

        for n in graph.neighbors_directed(v.id(), petgraph::Outgoing) {
            let nw = graph.node_weight(n).unwrap();
            let nid = nw.numerical_id;
            let npid = pid_array[nid];
            //println!("{:?}, {}", nid, npid);
        }

        //println!();
    }

    let exec_time = *next_idle_cycle.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let total_cpu_time = next_idle_cycle.iter().sum();
    let speedup = (total_cpu_time as f64) / (exec_time as f64);

    //println!("(exec_time, cpu_time, speedup) = ({}, {}, {})", exec_time, total_cpu_time, speedup);

    ExecutionInfo{exec_time, total_cpu_time, speedup}
}

#[allow(dead_code)]
#[derive(Debug,Clone,Copy)]
struct ExecutionUnit {
    remaining_work: usize,
    pid: usize,
    current_task_node: Option<petgraph::graph::NodeIndex<usize>>
}

fn get_empty_ready_task_queues(num_cores: usize) -> Vec<Vec<ExecutionUnit>> {
    vec![vec![]; num_cores]
}

fn get_empty_core_states(num_cores: usize) -> Vec<Option<ExecutionUnit>> {
    vec![None; num_cores]
}

fn get_num_pending_deps(graph: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>) -> Vec<usize>{
    let mut num_pending_deps = vec![0; graph.node_count()];

    for n in graph.node_references().enumerate() {
        let num_deps = graph.neighbors_directed(n.1.0, petgraph::Incoming).count();
        //println!("{:?}, num_deps: {}", n.1, num_deps);

        num_pending_deps[n.0] = num_deps;
    }

    num_pending_deps
}

// TODO: Avoid allocating data for dep_counts every time
//       this function is called.
fn get_indices_of_free_tasks(dep_counts: &Vec<usize>) -> Vec<usize> {
    let mut res = vec![];

    for (i, v) in dep_counts.into_iter().enumerate() {
        if *v == 0 {
            res.push(i);
        }
    }

    //println!("Indices of free tasks: {:?}", res);

    res
}

const TASK_SIZE: usize = 2;
const COMM_PENALTY: usize = 1;

fn move_free_tasks_to_ready_queues(free_task_indices: &Vec<usize>, ready_queues: &mut Vec<Vec<ExecutionUnit>>, pid_array: &[usize], g: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>) {
    let node_indices: Vec<(petgraph::prelude::NodeIndex<usize>, &NodeInfo)> = g.node_references().collect();

    for i in free_task_indices {
        let pid = pid_array[*i];

        ready_queues[pid].push(ExecutionUnit{remaining_work: TASK_SIZE, pid, current_task_node: Some(node_indices[*i].0)});
        //ready_queues[pid].push(ExecutionUnit{remaining_work: TASK_SIZE, pid, current_task_node: Some(node_indices[*i].0)});
    }
}

// TODO: Let this automatically update free_task_indices, to avoid recomputation
//
// This returns the smallest number of cycles that will take to retire at least one more task, if
// possible.
// TODO: Let it update dep_counts in a way that is compatible with the changing number of nodes in
// the graph
fn retire_finished_tasks(g: &mut petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, core_states: &mut Vec<Option<ExecutionUnit>>, dep_counts: &mut Vec<usize>) -> Option<usize> {
    let mut min_step_for_more_retirements = usize::MAX;

    //println!("[retire_finished_tasks]: Before: {:?}", core_states);
    //println!("[retire_finished_tasks]: Graph before: {:?}", g);

    // We reborrow here just to avoid moving
    // core_states to the score of the for,
    // which would prevent us from using it
    // down later.
    for e in &mut *core_states {
        if e.is_some() {
            let e_unwrapped = e.unwrap();
            let remaining_work = e_unwrapped.remaining_work;

            if remaining_work == 0 {
                let v = e_unwrapped.current_task_node.unwrap();
                
                for n in g.neighbors_directed(v, petgraph::Outgoing) {
                    let nw = g.node_weight(n).unwrap();
                    let nid = nw.numerical_id;
                    //dep_counts[nid] = dep_counts[nid] - 1;
                }
                
                *e = None;

                g.remove_node(v);
            } else {
                if remaining_work < min_step_for_more_retirements {
                    min_step_for_more_retirements = remaining_work;
                }
            }
        }
    }

    //println!("[retire_finished_tasks]: After: {:?}", core_states);
    //println!("[retire_finished_tasks]: Graph after: {:?}", g);

    if min_step_for_more_retirements < usize::MAX {
        return Some(min_step_for_more_retirements);
    } else {
        return None;
    }
}

fn advance_simulation(step_size: usize, core_states: &mut Vec<Option<ExecutionUnit>>) {
    //println!("[advance_simulation]: step_size = {}", step_size);
    //println!("Before: {:?}", core_states);

    for s_wrapped in &mut *core_states {
        if s_wrapped.is_some() {
            let s = s_wrapped.unwrap();
            let mut updated = s.clone();
            updated.remaining_work -= step_size;
            s_wrapped.replace(updated);
        }
    }

    //println!("After: {:?}", core_states);
}

fn feed_idle_cores(ready_queues: &mut Vec<Vec<ExecutionUnit>>, core_states: &mut Vec<Option<ExecutionUnit>>) {
    let mut i = 0;
    for s_wrapped in &mut *core_states {
        if s_wrapped.is_none() {
            let w = ready_queues[i].pop();
            *s_wrapped = w;
        }

        i += 1;
    }
}

fn evaluate_execution_time_and_speedup(original_graph: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, pid_array: &[usize]) -> ExecutionInfo {
    let mut ready_queues = get_empty_ready_task_queues(MAX_NUMBER_OF_PARTITIONS);
    let mut core_states = get_empty_core_states(MAX_NUMBER_OF_PARTITIONS);

    let mut g = original_graph.clone();
    let mut current_time = 0;
    let total_cpu_time = original_graph.node_count() * TASK_SIZE;

    while g.node_count() > 0 {
        // TODO-PERFORMANCE: Avoid parsing the whole graph for detecting 0-dep tasks at every
        // iteration
        let mut dep_counts = get_num_pending_deps(&g);
        let free_task_indices = get_indices_of_free_tasks(&dep_counts);
        //move_free_tasks_to_ready_queues(&free_task_indices, &mut ready_queues, pid_array, &g);
        move_free_tasks_to_ready_queues(&free_task_indices, &mut ready_queues, pid_array, &g);
        let min_step_for_more_retirements = retire_finished_tasks(&mut g, &mut core_states, &mut dep_counts).unwrap_or(0);
        feed_idle_cores(&mut ready_queues, &mut core_states);
        advance_simulation(min_step_for_more_retirements, &mut core_states);
        current_time += min_step_for_more_retirements;
    }

    return ExecutionInfo{exec_time: current_time, total_cpu_time, speedup: (total_cpu_time as f64 / (current_time as f64))};
}

#[allow(dead_code)]
fn test_metaheuristics_03(num_iter: usize) {

    let mut report = Vec::with_capacity(20);
    let start = Instant::now();
    let g = gen_random_digraph(true, 16, Some(64), 32, Some(32), Some(8));
    println!("Time to generate random graph: {:?}", start.elapsed());

    const TEMP_FILE_NAME: &str = "metaheuristics_evolution.csv";

    let mut _f = File::create(TEMP_FILE_NAME).unwrap();
    const POP_SIZE: usize = 32;
    const NUM_ITER: u64 = 1000;

    let mut best_surprise_per_algo = HashMap::new();

    for _ in 0..num_iter {
        /*
        let start = Instant::now();

        //run_algo!(Fa::default(), report, g);
        let _s = Solver::build(Fa::default(), MyFunc{graph: &g})
            .task(|ctx| ctx.gen == NUM_ITER)
            .pop_num(POP_SIZE)
            .callback(|ctx| report.push(ctx.best_f))
            .solve()
            .unwrap();

        let duration = start.elapsed();
        println!("Elapsed time (Firefly algorithm): {:?}", duration);
        println!("Time per iteration: {:?}", duration / (NUM_ITER as u32));

        //write_report_and_clear("Firefly", &mut report, &mut _f);

        let mut algo_best = *best_surprise_per_algo.entry("Firefly").or_insert(f64::MAX);

        if _s.best_fitness() < algo_best {
            algo_best = _s.best_fitness();
            visualize_graph(&g, Some(&_s.result()), Some(format!("firefly_{}_{}_{}", POP_SIZE, NUM_ITER, algo_best)));
        }
        */

        let start = Instant::now();

        let _s = Solver::build(De::default(), MyFunc{graph: &g})
            .task(|ctx| ctx.gen == NUM_ITER)
            .pop_num(POP_SIZE)
            .callback(|ctx| report.push(ctx.best_f))
            .solve()
            .unwrap();

        let duration = start.elapsed();
        println!("Elapsed time (Differential Evolution): {:?}", duration);
        println!("Time per iteration: {:?}", duration / (NUM_ITER as u32));

        //write_report_and_clear("Differential Evolution", &mut report, &mut _f);

        let mut algo_best = *best_surprise_per_algo.entry("Differential Evolution").or_insert(f64::MAX);

        if _s.best_fitness() < algo_best {
            algo_best = _s.best_fitness();
            let start = Instant::now();
            visualize_graph(&g, Some(&_s.result()), Some(format!("differential_evolution_{}_{}_{}", POP_SIZE, NUM_ITER, algo_best)));
            println!("Time to generate graph visualization: {:?}", start.elapsed());
            let start = Instant::now();
            println!("{:?}", evaluate_execution_time_and_speedup(&g, &_s.result()));
            println!("Time to simulate execution: {:?}", start.elapsed());
        }

        /*
        let _s = Solver::build(Pso::default(), MyFunc{graph: &g})
            .task(|ctx| ctx.gen == NUM_ITER)
            .pop_num(POP_SIZE)
            .callback(|ctx| report.push(ctx.best_f))
            .solve()
            .unwrap();

        write_report_and_clear("Particle Swarm Optimization", &mut report, &mut _f);

        let _s = Solver::build(Rga::default(), MyFunc{graph: &g})
            .task(|ctx| ctx.gen == NUM_ITER)
            .pop_num(POP_SIZE)
            .callback(|ctx| report.push(ctx.best_f))
            .solve()
            .unwrap();

        write_report_and_clear("Real-Coded Genetic Algorithm", &mut report, &mut _f);

        let _s = Solver::build(Tlbo::default(), MyFunc{graph: &g})
            .task(|ctx| ctx.gen == NUM_ITER)
            .pop_num(POP_SIZE)
            .callback(|ctx| report.push(ctx.best_f))
            .solve()
            .unwrap();

        write_report_and_clear("Teaching Learning Based Optimization", &mut report, &mut _f);
        */
    }

    let start = Instant::now();
    gen_scatter_evolution(TEMP_FILE_NAME, "test");
    println!("Time to generate surprise evolution graph: {:?}", start.elapsed());
}

fn main() {
    //gen_sample_graph_image();
    //test_histogram_01();
    //test_metaheuristics_01();
    //test_metaheuristics_02();
    test_metaheuristics_03(1);
    
    /*
    for i in [1,2,3] {
        println!("{i}");
    }
    */
}
