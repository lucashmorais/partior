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
use metaheuristics_nature::ndarray::{Array2, ArrayBase, OwnedRepr, Dim};

use metaheuristics_nature::{Rga, Fa, Pso, De, Tlbo, Solver};
use metaheuristics_nature::tests::TestObj;

const MAX_NUMBER_OF_PARTITIONS: usize = 4;
const MAX_NUMBER_OF_NODES: usize = 64;

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
    // a_raw: large
    // b_raw: small
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

fn calculate_pid_array_min_max_distance(pid_array: Option<&[usize]>, num_nodes: usize) -> f64 {
    let mut num_items_per_pid: HashMap<usize, usize> = HashMap::new();
    let pid_array = pid_array.unwrap();

    for pid in 0..num_nodes {
        num_items_per_pid.insert(pid, 1);
    }

    for i in 0..num_nodes {
        let pid = &pid_array[i];
        let current_num = num_items_per_pid.get(pid).unwrap_or(&0);
        num_items_per_pid.insert(*pid, current_num + 1);
    }

    let mut min = usize::MAX;
    let mut max = usize::MIN;
    for v in num_items_per_pid.values() {
        if *v < min {
            min = *v;
        }

        if *v > max {
            max = *v;
        }
    }

    (max - min) as f64 / MAX_NUMBER_OF_NODES as f64
}

fn expand_compact_pid(compact_pid_array: &[usize], pid_array: &mut [usize]) {
    let mut partition_sizes = [0; MAX_NUMBER_OF_PARTITIONS];
    let num_partition_elements = MAX_NUMBER_OF_PARTITIONS - 1;

    let mut remaining_nodes = MAX_NUMBER_OF_NODES;
    let mut max_i = 0;
    for i in 0..num_partition_elements {
        // The `1` ensures that every partition has a different
        // size, which is useful for reducing the search space
        let size = partition_sizes[i] + (compact_pid_array[i] + 1);

        if remaining_nodes >= size {
            partition_sizes[i+1] = size;
            remaining_nodes -= size;
        } else {
            if partition_sizes[i] < remaining_nodes {
                partition_sizes[i+1] = remaining_nodes;
            } else {
                partition_sizes[i] += remaining_nodes;
            }
            remaining_nodes = 0;
        }

        if partition_sizes[i] > partition_sizes[max_i] {
            max_i = i;
        }

        if partition_sizes[i+1] > partition_sizes[max_i] {
            max_i = i+1;
        }
    }

    partition_sizes[max_i] += remaining_nodes;

    //println!("{:?}", partition_sizes);

    for i in 1..max_i {
        if partition_sizes[i] <= partition_sizes[i - 1] {
            println!("{:?}", partition_sizes);
            assert!(false);
        }
    }
    //println!("{:?}", partition_sizes);
    
    let offset = num_partition_elements;
    for i in 0..MAX_NUMBER_OF_NODES {
        let base_pid_step = compact_pid_array[offset + i];

        for step in 0..MAX_NUMBER_OF_PARTITIONS - 1 {
            let target_pid = (step + base_pid_step) % MAX_NUMBER_OF_PARTITIONS;
            if partition_sizes[target_pid] > 0 {
                partition_sizes[target_pid] -= 1;
                pid_array[i] = target_pid;
            }
        }
    }
    //println!("{:?}", pid_array);
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
    
    //surprise = surprise + calculate_pid_array_min_max_distance(pid_array, num_nodes) * 10.;

    surprise 
}

// Implemented according to Chakraborty, Tanmoy, et al. "On the permanence of vertices in network communities."
// Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014.
// https://doi.org/10.1145/2623330.2623707
fn node_permanence(nid: Option<usize>, v: Option<NodeIndex<usize>>, original_graph: &Graph<NodeInfo, usize, petgraph::Directed, usize>, finalized_core_placements: &[Option<usize>], pid_array: &[usize]) -> f64 {
    let nid_unwrapped = nid.unwrap_or_default();
    let mut update_nid = false;

    let v = if v.is_some() {
        update_nid = true;
        v.unwrap()
    } else {
        assert!(nid.is_some());
        original_graph.node_references().find(|x| x.1.numerical_id == nid_unwrapped).unwrap().id()
    };

    let nid_unwrapped = if update_nid {
        original_graph.node_weight(v).unwrap().numerical_id
    } else {
        nid_unwrapped
    };

    let v_pid = pid_array[nid_unwrapped];

    let mut cluster_pulls: HashMap<usize, usize> = HashMap::new();

    let mut v_degree = 0;
    let mut internal_pull = 0;
    for n in original_graph.neighbors(v) {
        v_degree += 1;

        let n_nid = original_graph.node_weight(n).unwrap().numerical_id;
        let n_pid = finalized_core_placements[n_nid].unwrap_or(pid_array[n_nid]);

        if n_pid != v_pid {
            let current_cluster_pull = cluster_pulls.entry(n_pid).or_insert(0);
            *current_cluster_pull += 1;
        } else {
            internal_pull += 1;
        }
    }

    if v_degree == 0 {
        return 0.;
    }

    let mut max_external_pull = 0;
    //let mut external_cluster_with_maximum_pull = 0;

    for (_pid, pull) in cluster_pulls {
        if pull > max_external_pull {
            max_external_pull = pull;
            //external_cluster_with_maximum_pull = _pid;
        }
    }

    let mut num_nodes_in_v_cluster = 0;
    let mut num_edges_in_v_cluster = 0;

    for k in original_graph.node_references() {
        let k_nid = original_graph.node_weight(k.id()).unwrap().numerical_id;
        let k_pid = finalized_core_placements[k_nid].unwrap_or(pid_array[k_nid]);

        if k_pid == v_pid {
            num_nodes_in_v_cluster += 1;

            for nk in original_graph.neighbors_directed(k.id(), petgraph::Incoming) {
                let nk_nid = original_graph.node_weight(nk).unwrap().numerical_id;
                let nk_pid = finalized_core_placements[nk_nid].unwrap_or(pid_array[nk_nid]);

                // We do not take edges involving the query node into account
                // for calculating internal cluster coefficient
                if nk_pid == k_pid && k_nid != nid_unwrapped && nk_nid != nid_unwrapped {
                    num_edges_in_v_cluster += 1;
                }
            }
        }
    }
    // We do not take the query node into account
    // for calculating internal cluster coefficient
    num_nodes_in_v_cluster -= 1;

    let max_edges_in_v_cluster = num_nodes_in_v_cluster * (num_nodes_in_v_cluster - 1) / 2;
    let internal_cluster_coefficient = if num_edges_in_v_cluster > 2 && num_nodes_in_v_cluster > 2 {num_edges_in_v_cluster as f64 / max_edges_in_v_cluster as f64} else {0.};

    if max_external_pull > 0 {
        return (internal_pull as f64 / max_external_pull as f64) / (v_degree as f64) - (1. - internal_cluster_coefficient);
    } else {
        return internal_pull as f64 / v_degree as f64 - (1. - internal_cluster_coefficient);
    }
}

fn calculate_permanence(original_graph: &Graph<NodeInfo, usize, petgraph::Directed, usize>, finalized_core_placements: &[Option<usize>], pid_array: &[usize]) -> f64 {
    //original_graph.node_references().map(|v| node_permanence(None, Some(v.0), original_graph, finalized_core_placements, pid_array)).fold(0., |acc, x| acc + x) / original_graph.node_count() as f64

    original_graph.node_references().map(|v| node_permanence(None, Some(v.0), original_graph, finalized_core_placements, pid_array)).fold(0., |acc, x| acc + x) / original_graph.node_count() as f64 - calculate_pid_array_min_max_distance(Some(pid_array), original_graph.node_count())
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

    let res = (num_complete_groups * nodes_per_complete_group * (num_nodes - nodes_per_complete_group) + nodes_in_incomplete_group * (num_nodes - nodes_in_incomplete_group)) / 2;

    println!("[max_edges_for_given_parallelism]: (num_complete_groups, nodes_per_complete_group, nodes_in_incomplete_group, max_edges) = ({}, {}, {}, {})",
        num_complete_groups,
        nodes_per_complete_group,
        nodes_in_incomplete_group,
        res);

    res
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
            //b = rng.gen_range(first_option..first_option+5);
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

fn gen_speedup_bars(csv_path_with_suffix: &'static str, image_output_without_suffix: &'static str) {
    let _dot_proc_output = Command::new("src/speedup_graphs.py").arg(csv_path_with_suffix).arg(image_output_without_suffix).output().unwrap();
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

fn visualize_graph(g: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, pid_array: Option<&[Option<usize>]>, output_name: Option<String>) {
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
            let new_pid = pid_array[w.numerical_id].unwrap();
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

/*
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
*/

#[allow(dead_code)]
fn gen_sample_graph_image() {
    //let g = gen_sample_graph();
    let mut g = gen_random_digraph(true, 16, Some(16), 0, None, None);
    let new_graph = set_random_partitions_and_visualize_graph(&mut g, MAX_NUMBER_OF_PARTITIONS);
    visualize_graph(&new_graph, None, None);
}

/*
#[allow(dead_code)]
fn test_histogram_01() {
    let g = gen_random_digraph(true, 16, Some(16), 0, None, None);
    evaluate_multiple_random_clusterings(&g, MAX_NUMBER_OF_PARTITIONS, 100000000, true);
}
*/

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

// We need an explicit lifetime annotation here
// to ensure that the graph lives for as long as
// the solver itself lives. If the solver outlived
// the graph, it would end up referencing an
// invalid memory region once if we tried to
// go through the graph again.
//
// If this struct actually held the ownership of
// a graph rather than merely a reference to one,
// this would not be necessary, since the graph
// would only be dropped from memory once the
// solver is also dropped.
struct BaseSolver<'a> {
    graph: &'a petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>,
    core_bound: &'a [[f64; 2]]
}

impl Bounded for BaseSolver<'_> {
    fn bound(&self) -> &[[f64; 2]] {
        self.core_bound
        //&[[0., MAX_NUMBER_OF_PARTITIONS as f64]; MAX_NUMBER_OF_NODES]
    }
}

struct CompactSolver<'a> {
    graph: &'a petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>,
    core_bound: &'a [[f64; 2]]
}

impl Bounded for CompactSolver<'_> {
    fn bound(&self) -> &[[f64; 2]] {
        self.core_bound
        //&[[0., MAX_NUMBER_OF_PARTITIONS as f64]; MAX_NUMBER_OF_NODES]
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
        int_arr[i] = float_arr[i] as usize % MAX_NUMBER_OF_PARTITIONS;
    }

    int_arr
}

// Here we use a lifetime annotation just to
// tell the compiler that this trait implementation
// is useful for BaseSolver instances with any
// lifetime specification.
impl ObjFactory for BaseSolver<'_> {
    //type Product = [usize; MAX_NUMBER_OF_NODES];
    type Product = Vec<usize>;
    type Eval = f64;

    fn produce(&self, xs: &[f64]) -> Self::Product {
        //round_float_array(xs)
        return floor_float_array(xs).into();
    }

    //fn evaluate(&self, x: [usize; MAX_NUMBER_OF_NODES]) -> Self::Eval {
    fn evaluate(&self, x: Vec<usize>) -> Self::Eval {
        //400. + (-original_calculate_surprise(&self.graph, Some(&x))).log10()
        //1000.0 + calculate_surprise(&self.graph, Some(&x))
        
        calculate_surprise(&self.graph, Some(&x))
        //1. - calculate_permanence(&self.graph, &[None; MAX_NUMBER_OF_NODES], &x)
    }
}

impl ObjFactory for CompactSolver<'_> {
    //type Product = [usize; MAX_NUMBER_OF_NODES];
    type Product = [usize; MAX_NUMBER_OF_PARTITIONS - 1 + MAX_NUMBER_OF_NODES];
    type Eval = f64;

    fn produce(&self, xs: &[f64]) -> Self::Product {
        const C_SIZE: usize = MAX_NUMBER_OF_PARTITIONS - 1 + MAX_NUMBER_OF_NODES;
        let mut compact_pid_array = [0; C_SIZE];
        
        for i in 0..C_SIZE {
            compact_pid_array[i] = xs[i] as usize;
        }

        compact_pid_array
    }

    //fn evaluate(&self, x: [usize; MAX_NUMBER_OF_NODES]) -> Self::Eval {
    fn evaluate(&self, x: [usize; MAX_NUMBER_OF_PARTITIONS - 1 + MAX_NUMBER_OF_NODES]) -> Self::Eval {
        //400. + (-original_calculate_surprise(&self.graph, Some(&x))).log10()
        //1000.0 + calculate_surprise(&self.graph, Some(&x))
        
        let mut pid_array = [0; MAX_NUMBER_OF_NODES];
        expand_compact_pid(&x, &mut pid_array);
        calculate_surprise(&self.graph, Some(&pid_array))
        //1. - calculate_permanence(&self.graph, &[None; MAX_NUMBER_OF_NODES], &x)
    }
}

/*
#[allow(dead_code)]
// https://docs.rs/metaheuristics-nature/8.0.4/metaheuristics_nature/trait.ObjFunc.html
fn test_metaheuristics_02() {
    let mut report = Vec::with_capacity(20);
    let g = gen_random_digraph(true, 16, Some(32), 64, Some(96), None);

// Build and run the solver
    //let s = Solver::build(Rga::default(), BaseSolver{graph: &g})
    //let s = Solver::build(Pso::default(), BaseSolver{graph: &g})
    //let s = Solver::build(Tlbo::default(), BaseSolver{graph: &g})

    //let s = Solver::build(De::default(), BaseSolver{graph: &g})
    let s = Solver::build(Fa::default(), BaseSolver{graph: &g, finalized_core_placements: None})
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
*/

fn write_report_and_clear (name: &str, rep: &mut Vec<f64>, f: &mut File) {
    //f.write(format!("#ALG# {}\n", name).as_bytes()).unwrap();
    let num_elements = rep.len();

    //for h in &mut *rep {
    for i in 0..num_elements {
        let h = rep[i];
        f.write(format!("{name},{i},{h},fitness_test\n").as_bytes()).unwrap();
    }
    //f.write(format!("\n").as_bytes()).unwrap();

    rep.clear();
}

// TODO: Finish this
// Ref: https://doc.rust-lang.org/reference/macros-by-example.html
macro_rules! run_algo {
    ($t:expr, $report:ident, $g:ident) => {
        let _s = Solver::build($t, BaseSolver{graph: &$g, finalized_core_placements: None})
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
    speedup: f64,
    num_misses: usize,
    num_tasks_stolen: usize
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

#[allow(dead_code)]
#[derive(Debug,Clone,Copy)]
struct ExecutionUnit {
    remaining_work: usize,
    pid: usize,
    current_task_node: Option<usize>
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
        //println!("(n.0, n.1, num_deps) = ({:?}, {:?}, {:?})", n.0, n.1, num_deps);

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

const TASK_SIZE: usize = 20;
const COMM_PENALTY: usize = 1;
const SHARING_THRESHOLD: usize = 0;
//const SHARING_THRESHOLD: usize = usize::MAX;

//TODO-PERFORMANCE: AVOID RECALCULATION
fn move_free_tasks_to_ready_queues(
        ready_queues: &mut Vec<Vec<ExecutionUnit>>,
        pid_array: &[usize],
        g: &mut petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>,
        original_graph: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>,
        task_was_sent_to_ready_queue: &mut [bool]
    ) {
    let dep_counts = get_num_pending_deps(&g);
    let free_task_indices = &get_indices_of_free_tasks(&dep_counts);

    for i in free_task_indices {
        let nid = g.raw_nodes()[*i].weight.numerical_id;
        let v = original_graph.node_references().find(|x| x.1.numerical_id == nid).unwrap().id();
        //println!("This is (v, v.index()): ({:?}, {:?})", v, v.index());

        //let pid = pid_array[*i];
        let pid = pid_array[v.index()];

        if !task_was_sent_to_ready_queue[nid] {
            let e = ExecutionUnit{remaining_work: TASK_SIZE, pid, current_task_node: Some(nid)};

            //println!("[move_free_tasks_to_ready_queues]: e = {:?}", e);

            ready_queues[pid].push(e);
            task_was_sent_to_ready_queue[nid] = true;
        }
    }

    /*
     * THIS ABSOLUTELY CANNOT BE HERE,
     * BECAUSE ANY DEPENDENCE IS ONLY
     * FULFILLED AFTER THE OWNING TASK
     * HAS FINISHED EXECUTING AT THE
     * CORE.
    for nid in pending_removals {
        let v = g.node_references().find(|x| x.1.numerical_id == nid).unwrap().id();
        g.remove_node(v);
    }
    */
}

// TODO: Let this automatically update free_task_indices, to avoid recomputation
//
// This returns the smallest number of cycles that will take to retire at least one more task, if
// possible.
// TODO: Let it update dep_counts in a way that is compatible with the changing number of nodes in
// the graph
fn retire_finished_tasks(g: &mut petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, core_states: &mut Vec<Option<ExecutionUnit>>, pid_array_option: Option<&mut [usize]>) -> Option<usize> {
    let mut min_step_for_more_retirements = usize::MAX;
    let apply_immediate_successor;
    let pid_array;

    if pid_array_option.is_some() {
        apply_immediate_successor = true;
        pid_array = pid_array_option.unwrap();
    } else {
        apply_immediate_successor = false;
        pid_array = &mut [];
    }

    // We reborrow here just to avoid moving
    // core_states to the score of the for,
    // which would prevent us from using it
    // down later.
    //println!("[retire_finished_tasks]: core_states before: {:?}", core_states);
    for (_, e) in core_states.iter_mut().enumerate() {
        if e.is_some() {
            let e_unwrapped = e.unwrap();
            let remaining_work = e_unwrapped.remaining_work;

            if remaining_work == 0 {
                let v_id = e.unwrap().current_task_node.unwrap();
                let v = g.node_references().find(|x| x.1.numerical_id == v_id).unwrap().id();

                if apply_immediate_successor {
                    let v_pid = pid_array[v_id];
                    for n in g.neighbors_directed(v, petgraph::Outgoing) {
                        let n_id = g.node_weight(n).unwrap().numerical_id;
                        pid_array[n_id] = v_pid;
                    }
                }

                g.remove_node(v);

                *e = None;
            } else {
                if remaining_work < min_step_for_more_retirements {
                    min_step_for_more_retirements = remaining_work;
                }
            }
        }
    }
    //println!("[retire_finished_tasks]: core_states after: {:?}", core_states);


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

fn get_task_from_fullest_queue(ready_queues: &mut Vec<Vec<ExecutionUnit>>) -> Option<ExecutionUnit> {
    let mut max_num_elements = 0;
    let mut index_max_elements = 0;

    //println!("[get_task_from_fullest_queue]: (index_max_elements, max_num_elements) = ({}, {})", index_max_elements, max_num_elements);

    for (i, q) in ready_queues.iter().enumerate() {
        let num_elements = q.len();
        if num_elements > max_num_elements {
            max_num_elements = num_elements;
            index_max_elements = i;
        }
    }

    if ready_queues[index_max_elements].len() > SHARING_THRESHOLD {
        return ready_queues[index_max_elements].pop();
    }

    None
}

fn num_foreign_incoming_edges(nid: usize, original_graph: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, target_core: usize, finalized_core_placements: &mut [Option<usize>]) -> usize {
    let original_v = original_graph.node_references().find(|x| x.1.numerical_id == nid).unwrap().id();

    let mut num_foreign_edges = 0;
    for n in original_graph.neighbors_directed(original_v, petgraph::Incoming) {
        let n_nid = original_graph.node_weight(n).unwrap().numerical_id;
        let n_finalized_core = finalized_core_placements[n_nid].unwrap();

        if n_finalized_core != target_core {
            num_foreign_edges += 1;
            //println!("[num_foreign_incoming_edges]: (nid, n_nid, target_core, n_finalized_core) = ({}, {}, {}, {})", nid, n_nid, target_core, n_finalized_core);
        }
    }

    num_foreign_edges
}

fn num_native_edges(nid: usize, original_graph: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, finalized_core_placements: &mut [Option<usize>]) -> usize {
    let original_v = original_graph.node_references().find(|x| x.1.numerical_id == nid).unwrap().id();
    let original_pid = original_graph.node_weight(original_v).unwrap().partition_id;

    let mut num_native_edges = 0;

    for n in original_graph.neighbors_directed(original_v, petgraph::Incoming) {
        let n_nid = original_graph.node_weight(n).unwrap().numerical_id;
        let n_finalized_core = finalized_core_placements[n_nid].unwrap();

        if n_finalized_core == original_pid {
            num_native_edges += 1;
            //println!("[num_native_incoming_edges]: (nid, n_nid, original_pid, n_finalized_core) = ({}, {}, {}, {})", nid, n_nid, original_pid, n_finalized_core);
        }
    }

    for n in original_graph.neighbors_directed(original_v, petgraph::Outgoing) {
        //let n_nid = original_graph.node_weight(n).unwrap().numerical_id;
        let n_pid = original_graph.node_weight(n).unwrap().partition_id;

        if n_pid == original_pid {
            num_native_edges += 1;
            //println!("[num_native_incoming_edges]: (nid, n_nid, original_pid, n_finalized_core) = ({}, {}, {}, {})", nid, n_nid, original_pid, n_finalized_core);
        }
    }

    num_native_edges
}

const LIMIT_LATE_STEALING: bool = false;

fn num_edges_to_target_core(nid: usize, original_graph: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, target_core: usize, finalized_core_placements: &mut [Option<usize>]) -> usize {
    let original_v = original_graph.node_references().find(|x| x.1.numerical_id == nid).unwrap().id();

    let mut num_edges_to_target = 0;
    for n in original_graph.neighbors_directed(original_v, petgraph::Incoming) {
        let n_nid = original_graph.node_weight(n).unwrap().numerical_id;
        let n_finalized_core = finalized_core_placements[n_nid].unwrap();

        if n_finalized_core == target_core {
            num_edges_to_target += 1;
        }
    }

    for n in original_graph.neighbors_directed(original_v, petgraph::Outgoing) {
        //let n_nid = original_graph.node_weight(n).unwrap().numerical_id;
        let n_pid = original_graph.node_weight(n).unwrap().partition_id;

        if n_pid == target_core {
            num_edges_to_target += 1;
        }
    }

    num_edges_to_target
}

fn get_task_with_lowest_cluster_degree(ready_queues: &mut Vec<Vec<ExecutionUnit>>, original_graph: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, finalized_core_placements: &mut [Option<usize>]) -> Option<ExecutionUnit> {
    let mut lowest = usize::MAX;
    let mut li = 0;
    let mut lj = 0;

    for (i, q) in ready_queues.into_iter().enumerate() {
        for (j, e) in q.into_iter().enumerate() {
            let nid = e.current_task_node.unwrap();

            // TODO: This restricts stealing of the last 10% of tasks to avoid impacting
            //       final execution time at the last moment.
            if !LIMIT_LATE_STEALING || nid < (0.85 * MAX_NUMBER_OF_NODES as f64) as usize {
                let cluster_degree = num_native_edges(nid, original_graph, finalized_core_placements);

                if cluster_degree < lowest {
                    lowest = cluster_degree;
                    li = i;
                    lj = j;
                }
            }
        }
    }

    if lowest < usize::MAX {
        return Some(ready_queues[li].remove(lj));
    }

    None
}

fn get_task_with_lowest_extra_expected_misses(ready_queues: &mut Vec<Vec<ExecutionUnit>>, original_graph: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, finalized_core_placements: &mut [Option<usize>], target_core: usize) -> Option<ExecutionUnit> {
    let mut lowest = usize::MAX;
    let mut li = 0;
    let mut lj = 0;

    for (i, q) in ready_queues.into_iter().enumerate() {
        for (j, e) in q.into_iter().enumerate() {
            let nid = e.current_task_node.unwrap();

            // TODO: This restricts stealing of the last 10% of tasks to avoid impacting
            //       final execution time at the last moment.
            if !LIMIT_LATE_STEALING || nid < (0.85 * MAX_NUMBER_OF_NODES as f64) as usize {
                let num_expected_misses = MAX_NUMBER_OF_NODES +  num_native_edges(nid, original_graph, finalized_core_placements) - num_edges_to_target_core(nid, original_graph, target_core, finalized_core_placements);
                // We add MAX_NUMBER_OF_NODES here just to avoid an underflow

                if num_expected_misses < lowest {
                    lowest = num_expected_misses;
                    li = i;
                    lj = j;
                }
            }
        }
    }

    if lowest < usize::MAX {
        return Some(ready_queues[li].remove(lj));
    }

    None
}

fn feed_idle_cores(ready_queues: &mut Vec<Vec<ExecutionUnit>>, core_states: &mut Vec<Option<ExecutionUnit>>, original_graph: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, finalized_core_placements: &mut [Option<usize>]) -> (usize, usize) {
    let mut num_misses = 0;
    let mut num_tasks_stolen = 0;

    for steal in [false, true] {
    //for steal in [false] {
        for i in 0..MAX_NUMBER_OF_PARTITIONS {
            let s_wrapped = &mut core_states[i];

            if s_wrapped.is_none() {
                //let vl = ready_queues[i].len();
                
                let mut w = None;
                if !steal {
                    w = ready_queues[i].pop();
                }

                //assert!(vl == 0 || (ready_queues[i].len() == vl - 1));
                //println!("[feed_idle_cores]: popped task: {:?}", w);
                
                if steal && w.is_none() {
                    //w = get_task_from_fullest_queue(ready_queues);
                    //w = get_task_with_lowest_cluster_degree(ready_queues, original_graph, finalized_core_placements);
                    w = get_task_with_lowest_extra_expected_misses(ready_queues, original_graph, finalized_core_placements, i);
                    if w.is_some() {
                        num_tasks_stolen += 1;
                    }
                }

                // This cannot be an else statement
                if w.is_some() {
                    let mut w_unwrapped = w.unwrap().clone();
                    let nid = w_unwrapped.current_task_node.unwrap();
                    finalized_core_placements[nid] = Some(i);
                    let task_num_misses = num_foreign_incoming_edges(nid, original_graph, i, finalized_core_placements);
                    w_unwrapped.remaining_work += COMM_PENALTY * task_num_misses;
                    num_misses += task_num_misses;

                    //println!("Feeding task {} to core {}", nid, i);

                    /*
                    if task_num_misses > 0 {
                        println!("Just caused {} cache misse(s). Stealing? {}", task_num_misses, steal);
                    }
                    */

                    w = Some(w_unwrapped);
                }

                *s_wrapped = w;
            }
        }
    }

    (num_misses, num_tasks_stolen)
}

fn all_ready_queues_are_empty(ready_queues: &Vec<Vec<ExecutionUnit>>) -> bool {
    for q in ready_queues {
        if q.len() > 0 {
            return false;
        }
    }

    true
}

fn all_cores_are_empty(core_states: &Vec<Option<ExecutionUnit>>) -> bool {
    for s in core_states {
        if s.is_some() {
            return false;
        }
    }

    true
}

fn update_ready_queues_with_new_pid_data(ready_queues: &mut Vec<Vec<ExecutionUnit>>, pid_array: &[usize]) {
    let mut all_tasks: Vec<ExecutionUnit> = vec![];

    for q in &mut *ready_queues {
        all_tasks.append(q);
    }

    for t in all_tasks {
        let nid = t.current_task_node.unwrap();
        let pid = pid_array[nid];
        
        ready_queues[pid].push(t);
    }
}

fn build_bound_vec(finalized_core_placements: &[Option<usize>]) -> Vec<[f64; 2]> {
    let mut bound_vec = vec![[0., MAX_NUMBER_OF_PARTITIONS as f64]; MAX_NUMBER_OF_NODES];

    for p in finalized_core_placements.into_iter().enumerate().filter(|(_, v)| v.is_some()) {
        let (i, k) = p;
        let k2 = k.unwrap() as f64;
        bound_vec[i] = [k2, k2];
    }

    bound_vec
}

const EPS: f64 = 0.000000000001;
fn build_bound_vec_for_compact_solver() -> Vec<[f64; 2]> {
    let mut bound_vec = vec![[0., MAX_NUMBER_OF_PARTITIONS as f64 - EPS]; MAX_NUMBER_OF_PARTITIONS - 1 + MAX_NUMBER_OF_NODES];

    for i in 0..(MAX_NUMBER_OF_PARTITIONS - 1) {
        bound_vec[i] = [0., MAX_NUMBER_OF_NODES as f64 + 1. - EPS];
    }

    bound_vec
}

fn evaluate_execution_time_and_speedup(original_graph: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, pid_array: &[usize], num_generations: usize, immediate_successor: bool) -> ([Option<usize>; MAX_NUMBER_OF_NODES], ExecutionInfo) {
    let mut ready_queues = get_empty_ready_task_queues(MAX_NUMBER_OF_PARTITIONS);
    let mut core_states = get_empty_core_states(MAX_NUMBER_OF_PARTITIONS);

    let mut g = original_graph.clone();
    let mut current_time = 0;
    let mut finalized_core_placements = [None; MAX_NUMBER_OF_NODES];
    let mut task_was_sent_to_ready_queue = [false; MAX_NUMBER_OF_NODES];
    let mut num_misses = 0;
    let mut num_tasks_stolen = 0;
    let total_cpu_time = original_graph.node_count() * TASK_SIZE;

    if immediate_successor {
        let mut pid_array = [0; MAX_NUMBER_OF_NODES];

        while g.node_count() > 0 || !all_ready_queues_are_empty(&ready_queues) || !all_cores_are_empty(&core_states) {

            //let core_bound = &build_bound_vec(&finalized_core_placements);

            //let _s = de_solve(&g, num_generations, 32, None, Some(&finalized_core_placements), false, core_bound);
            //let pid_array = &_s.result();
            //
            // TODO-PERFORMANCE: Avoid re-processing all ready tasks at every iteration
            update_ready_queues_with_new_pid_data(&mut ready_queues, &pid_array);

            // TODO-PERFORMANCE: Avoid parsing the whole graph for detecting 0-dep tasks at every iteration
            move_free_tasks_to_ready_queues(&mut ready_queues, &pid_array, &mut g, &original_graph, &mut task_was_sent_to_ready_queue);
            let (aux_num_misses, aux_num_tasks_stolen) = feed_idle_cores(&mut ready_queues, &mut core_states, &original_graph, &mut finalized_core_placements);
            num_misses += aux_num_misses;
            num_tasks_stolen += aux_num_tasks_stolen;
            let min_step_for_more_retirements = retire_finished_tasks(&mut g, &mut core_states, Some(&mut pid_array)).unwrap_or(0);
            advance_simulation(min_step_for_more_retirements, &mut core_states);
            current_time += min_step_for_more_retirements;
        }
    } else {
        while g.node_count() > 0 || !all_ready_queues_are_empty(&ready_queues) || !all_cores_are_empty(&core_states) {

            //let core_bound = &build_bound_vec(&finalized_core_placements);

            //let _s = de_solve(&g, num_generations, 32, None, Some(&finalized_core_placements), false, core_bound);
            //let pid_array = &_s.result();
            //
            // TODO-PERFORMANCE: Avoid re-processing all ready tasks at every iteration
            update_ready_queues_with_new_pid_data(&mut ready_queues, pid_array);

            // TODO-PERFORMANCE: Avoid parsing the whole graph for detecting 0-dep tasks at every iteration
            move_free_tasks_to_ready_queues(&mut ready_queues, pid_array, &mut g, &original_graph, &mut task_was_sent_to_ready_queue);
            let (aux_num_misses, aux_num_tasks_stolen) = feed_idle_cores(&mut ready_queues, &mut core_states, &original_graph, &mut finalized_core_placements);
            num_misses += aux_num_misses;
            num_tasks_stolen += aux_num_tasks_stolen;
            let min_step_for_more_retirements = retire_finished_tasks(&mut g, &mut core_states, None).unwrap_or(0);
            advance_simulation(min_step_for_more_retirements, &mut core_states);
            current_time += min_step_for_more_retirements;
        }
    }

    return (finalized_core_placements, ExecutionInfo{exec_time: current_time, total_cpu_time, speedup: (total_cpu_time as f64 / (current_time as f64)), num_misses, num_tasks_stolen});
}

struct RunConfig {
    probe_step_size: usize
}

fn de_solve<'a>(g: &'a petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, num_generations: usize, population_size: usize, report: Option<&mut Vec<f64>>, finalized_core_placements: Option<&'a [Option<usize>]>, verbose: bool, core_bound: &'a [[f64; 2]], config: &RunConfig, partial_solutions: &mut Vec<(usize, Vec<usize>)>) -> Solver<BaseSolver<'a>>{
    let start = Instant::now();

    //let core_bound = &[[0., MAX_NUMBER_OF_PARTITIONS as f64]; MAX_NUMBER_OF_NODES][..];

    if report.is_some() {
        let report = report.unwrap();

        //let _s = Solver::build(De::default().f(0.5), BaseSolver{graph: &g, core_bound})
        let _s = Solver::build(De::default(), BaseSolver{graph: &g, core_bound})
            .task(|ctx| ctx.gen == num_generations as u64)
            .pop_num(population_size)
            //.pool(|ctx, rng| Array2::from_shape_fn(ctx.pool_size(), |(_, s)| rng.range(ctx.func.bound_range(s))))
            /*
            .pool(|ctx, rng| Array2::from_shape_fn(ctx.pool_size(), |(i, j)| {
                if i < 2 {
                    return rng.range(ctx.func.bound_range(j))
                } else {
                    return 0.
                }
            }))
            */
            .callback(|ctx| {
                report.push(ctx.best_f);
                if ctx.gen % config.probe_step_size as u64 == 0 {
                    partial_solutions.push((ctx.gen as usize, ctx.result()));
                }
                
            })
            .solve()
            .unwrap();

        if verbose {
            let duration = start.elapsed();
            println!("Elapsed time (Differential Evolution): {:?}", duration);
            if num_generations > 0 {
                println!("Time per iteration: {:?}", duration / (num_generations as u32));
            }
        }

        //write_report_and_clear("Differential Evolution", &mut report, &mut _f);

        return _s;
    } else {
            let _s = Solver::build(De::default(), BaseSolver{graph: &g, core_bound})
            .task(|ctx| ctx.gen == num_generations as u64)
            .pop_num(population_size)
            //.pool(|ctx, rng| Array2::from_shape_fn(ctx.pool_size(), |(_, s)| rng.range(ctx.func.bound_range(s))))
            .pool(|ctx, rng| Array2::from_shape_fn(ctx.pool_size(), |(i, j)| {
                if i < 2 {
                    return rng.range(ctx.func.bound_range(j))
                } else {
                    return 0.
                }
            }))
            .solve()
            .unwrap();

        if verbose {
            let duration = start.elapsed();
            println!("Elapsed time (Differential Evolution): {:?}", duration);
            if num_generations > 0 {
                println!("Time per iteration: {:?}", duration / (num_generations as u32));
            }
        }

        return _s;
    }
}

fn de_compact_solve<'a>(g: &'a petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, num_generations: usize, population_size: usize, report: Option<&mut Vec<f64>>, finalized_core_placements: Option<&'a [Option<usize>]>, verbose: bool, core_bound: &'a [[f64; 2]]) -> Solver<CompactSolver<'a>>{
    let start = Instant::now();

    //let core_bound = &[[0., MAX_NUMBER_OF_PARTITIONS as f64]; MAX_NUMBER_OF_NODES][..];

    if report.is_some() {
        let report = report.unwrap();

        let _s = Solver::build(De::default(), CompactSolver{graph: &g, core_bound})
            .task(|ctx| ctx.gen == num_generations as u64)
            .pop_num(population_size)
            //.pool(|ctx, rng| Array2::from_shape_fn(ctx.pool_size(), |(_, s)| rng.range(ctx.func.bound_range(s))))
            /*
            .pool(|ctx, rng| Array2::from_shape_fn(ctx.pool_size(), |(i, j)| {
                if i < 2 {
                    return rng.range(ctx.func.bound_range(j))
                } else {
                    return 0.
                }
            }))
            */
            .callback(|ctx| report.push(ctx.best_f))
            .solve()
            .unwrap();

        if verbose {
            let duration = start.elapsed();
            println!("Elapsed time (Differential Evolution): {:?}", duration);
            if num_generations > 0 {
                println!("Time per iteration: {:?}", duration / (num_generations as u32));
            }
        }

        //write_report_and_clear("Differential Evolution", &mut report, &mut _f);

        return _s;
    } else {
        let _s = Solver::build(De::default(), CompactSolver{graph: &g, core_bound})
            .task(|ctx| ctx.gen == num_generations as u64)
            .pop_num(population_size)
            //.pool(|ctx, rng| Array2::from_shape_fn(ctx.pool_size(), |(_, s)| rng.range(ctx.func.bound_range(s))))
            .pool(|ctx, rng| Array2::from_shape_fn(ctx.pool_size(), |(i, j)| {
                if i < 2 {
                    return rng.range(ctx.func.bound_range(j))
                } else {
                    return 0.
                }
            }))
            .solve()
            .unwrap();

        if verbose {
            let duration = start.elapsed();
            println!("Elapsed time (Differential Evolution): {:?}", duration);
            if num_generations > 0 {
                println!("Time per iteration: {:?}", duration / (num_generations as u32));
            }
        }

        return _s;
    }
}

fn gen_lfr_like_graph(num_nodes: usize, num_edges: usize, mixing_coeff: f64, num_communities: usize, max_comm_size_difference: usize) -> Graph<NodeInfo, usize, petgraph::Directed, usize> {
    let mut rng = rand::thread_rng();

    let d = max_comm_size_difference;
    let min_l = (num_nodes as f64 / num_communities as f64).ceil() as usize - d;
    let max_l = num_nodes / num_communities;
    
    let l = rng.gen_range(min_l..=max_l);
    let h = l + d;
    let mut missing_nodes = num_nodes - l * num_communities;

    let mut community_sizes = vec![l; num_communities];

    while missing_nodes > 0 {
        let r = rng.gen_range(0..num_communities);

        if community_sizes[r] < h {
            community_sizes[r] += 1;
            missing_nodes -= 1;
        }
    }

    let mut base_pid_array: Vec<usize> = Vec::with_capacity(num_nodes);

    for (i, s) in community_sizes.into_iter().enumerate() {
        for _ in 0..s {
            base_pid_array.push(i);
        }
    }

    let mut missing_nodes = num_nodes;
    let mut pid_array: Vec<usize> = Vec::with_capacity(num_nodes);
    
    while missing_nodes > 0 {
        let i = rng.gen_range(0..missing_nodes);
        pid_array.push(base_pid_array.swap_remove(i));
        missing_nodes -= 1;
    }

    let mut g = GraphMap::<NodeInfo, usize, petgraph::Directed>::with_capacity(num_nodes, num_edges);
    assert!(g.node_count() == 0);

    for i in 0..num_nodes {
        g.add_node(NodeInfo{numerical_id: i, partition_id: pid_array[i]});
    }

    let nodes: Vec<NodeInfo> = g.nodes().collect();
    let mut missing_edges = num_edges;

    while missing_edges > 0 {
        let a = rng.gen_range(0..num_nodes - 1);
        let b = rng.gen_range(a+1..num_nodes);

        if g.contains_edge(nodes[a], nodes[b]) {
            continue;
        }
        
        let probability_to_add = if pid_array[a] != pid_array[b] {
            mixing_coeff
        } else {
            1. - mixing_coeff
        };

        if rng.gen_bool(probability_to_add) {
            g.add_edge(nodes[a], nodes[b], 1);            
            missing_edges -= 1;
        }
    }

    let g: petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize> = g.into_graph();
    g
}

#[allow(dead_code)]
fn test_metaheuristics_03(num_iter: usize) {

    let mut report = Vec::with_capacity(20);
    let start = Instant::now();

    let num_nodes = 32;
    let num_edges = 32;
    let min_parallelism = 16;
    let mixing_coeff = 0.;
    let num_communities = 8;
    let max_comm_size_difference = 0;

    //let g = gen_random_digraph(true, 16, Some(160), 32, Some(600), Some(32));
    //let g = gen_random_digraph(true, 16, Some(num_nodes), 32, Some(num_edges), Some(min_parallelism));
    let g = gen_lfr_like_graph(num_nodes, num_edges, mixing_coeff, num_communities, max_comm_size_difference);
    println!("Time to generate random graph: {:?}", start.elapsed());

    const TEMP_FILE_NAME: &str = "metaheuristics_evolution.csv";

    let mut _f = File::create(TEMP_FILE_NAME).unwrap();
    const POP_SIZE: usize = 64;

    let mut best_surprise_per_algo = HashMap::new();

    let mut average_speedups: Vec<f64> = vec![];
    let mut average_fitnesses: Vec<f64> = vec![];
    let mut average_permanences: Vec<f64> = vec![];
    let mut last_immediate_successor_speedup = 0.;

    let num_gen_options = [8000];
    for num_generations in num_gen_options {
        let mut speedup_sum = 0.0;
        let mut fitness_sum = 0.0;
        let mut permanence_sum = 0.0;

        for _ in 0..num_iter {
            /*
            let start = Instant::now();

            //run_algo!(Fa::default(), report, g);
            let _s = Solver::build(Fa::default(), BaseSolver{graph: &g})
                .task(|ctx| ctx.gen == num_generations)
                .pop_num(POP_SIZE)
                .callback(|ctx| report.push(ctx.best_f))
                .solve()
                .unwrap();

            let duration = start.elapsed();
            println!("Elapsed time (Firefly algorithm): {:?}", duration);
            println!("Time per iteration: {:?}", duration / (num_generations as u32));

            //write_report_and_clear("Firefly", &mut report, &mut _f);

            let mut algo_best = *best_surprise_per_algo.entry("Firefly").or_insert(f64::MAX);

            if _s.best_fitness() < algo_best {
                algo_best = _s.best_fitness();
                visualize_graph(&g, Some(&_s.result()), Some(format!("firefly_{}_{}_{}", POP_SIZE, num_generations, algo_best)));
            }
            */

            /*
            let test_finalized_inner = [0; MAX_NUMBER_OF_NODES - 50];
            let test_finalized: Option<&[usize]> = Some(&test_finalized_inner);
            let _s = de_solve(&g, num_generations, POP_SIZE, Some(&mut report), test_finalized);
            */

            //let mut mut_core_bound = vec![[2., 2.]; MAX_NUMBER_OF_NODES];
            //mut_core_bound[5] = [3., 3.];
            //let core_bound = &mut_core_bound;

            let core_bound = &vec![[0., MAX_NUMBER_OF_PARTITIONS as f64]; MAX_NUMBER_OF_NODES];
            let conf = RunConfig{probe_step_size: 100};
            let mut partial_solutions: Vec<(usize, Vec<usize>)> = vec![];
            let _s = de_solve(&g, num_generations, POP_SIZE, Some(&mut report), None, true, core_bound, &conf, &mut partial_solutions);

            let mut partial_speedups: Vec<(usize, f64)> = vec![];
            for i in 0..partial_solutions.len() {
                let partial_solution = &partial_solutions[i];
                let pid_array = &partial_solution.1;
                let (finalized_core_placements, execution_info) = evaluate_execution_time_and_speedup(&g, pid_array, 0, false);
                let speedup = execution_info.speedup;
                let num_gen = partial_solution.0;
                let fitness = report[i];

                let void_finalized = [None; MAX_NUMBER_OF_NODES];
                //let permanence = calculate_permanence(&g, &finalized_core_placements, pid_array);
                let permanence = calculate_permanence(&g, &void_finalized, pid_array);

                partial_speedups.push((num_gen, speedup));
                _f.write(format!("Differential Evolution,{},{},fitness_test,{},{},{},{},{},{}\n", num_gen, fitness, speedup, MAX_NUMBER_OF_PARTITIONS, num_nodes, num_edges, min_parallelism, permanence).as_bytes()).unwrap();
            }

            println!("{:?}", partial_speedups);

            let (_, immediate_successor_execution_info) = evaluate_execution_time_and_speedup(&g, &_s.result(), num_generations, true);
            last_immediate_successor_speedup = immediate_successor_execution_info.speedup;

            let start = Instant::now();
            //visualize_graph(&g, Some(&_s.result()), Some(format!("differential_evolution_{}_{}_{}", POP_SIZE, num_generations, algo_best)));
            println!("Time to generate graph visualization: {:?}", start.elapsed());
            let start = Instant::now();
            let (finalized_core_placements, execution_info) = evaluate_execution_time_and_speedup(&g, &_s.result(), num_generations, false);
            println!("Time to simulate execution: {:?}", start.elapsed());
            speedup_sum += execution_info.speedup;
            fitness_sum += _s.best_fitness();
            let start = Instant::now();
            permanence_sum += calculate_permanence(&g, &finalized_core_placements, &_s.result());
            println!("Time to calculate permanence: {:?}", start.elapsed());
            println!("{:?}", execution_info);
            println!("Single-shot surprise: {:?}", _s.best_fitness());
            println!("_s.result():\t\t\t{:?}", _s.result());
            println!("finalized_core_placements:\t{:?}", finalized_core_placements);


            let algo_best = *best_surprise_per_algo.entry("Differential Evolution").or_insert(f64::MIN);
            //if _s.best_fitness() < algo_best {
            if execution_info.speedup > algo_best {
                //best_surprise_per_algo.insert("Differential Evolution", _s.best_fitness());
                best_surprise_per_algo.insert("Differential Evolution", execution_info.speedup);
                let mut res: Vec<Option<usize>> = vec![];
                _s.result().into_iter().for_each(|v| res.push(Some(v)));

                visualize_graph(&g, Some(&res), Some(format!("differential_evolution_predicted_placement_{}_{}_{}", POP_SIZE, num_generations, _s.best_fitness())));
                visualize_graph(&g, Some(&finalized_core_placements), Some(format!("differential_evolution_final_placement_{}_{}_{}_{}", POP_SIZE, num_generations, _s.best_fitness(), execution_info.speedup)));
                println!("Exact speedup: {:.32}", execution_info.speedup);
            }

            /*
            let _s = Solver::build(Pso::default(), BaseSolver{graph: &g})
                .task(|ctx| ctx.gen == num_generations)
                .pop_num(POP_SIZE)
                .callback(|ctx| report.push(ctx.best_f))
                .solve()
                .unwrap();

            write_report_and_clear("Particle Swarm Optimization", &mut report, &mut _f);

            let _s = Solver::build(Rga::default(), BaseSolver{graph: &g})
                .task(|ctx| ctx.gen == num_generations)
                .pop_num(POP_SIZE)
                .callback(|ctx| report.push(ctx.best_f))
                .solve()
                .unwrap();

            write_report_and_clear("Real-Coded Genetic Algorithm", &mut report, &mut _f);

            let _s = Solver::build(Tlbo::default(), BaseSolver{graph: &g})
                .task(|ctx| ctx.gen == num_generations)
                .pop_num(POP_SIZE)
                .callback(|ctx| report.push(ctx.best_f))
                .solve()
                .unwrap();

            write_report_and_clear("Teaching Learning Based Optimization", &mut report, &mut _f);
            */
        }
        let average_speedup = speedup_sum / (num_iter as f64);
        let average_fitness = fitness_sum / (num_iter as f64);
        let average_permanence = permanence_sum / (num_iter as f64);

        average_speedups.push(average_speedup);
        average_fitnesses.push(average_fitness);
        average_permanences.push(average_permanence);
        //_f.write(format!("Differential Evolution,{},{},fitness_test,{},{},{},{},{},{}\n", num_generations, average_fitness, average_speedup, MAX_NUMBER_OF_PARTITIONS, num_nodes, num_edges, min_parallelism, average_permanence).as_bytes()).unwrap();
    }

    gen_speedup_bars(TEMP_FILE_NAME, "speedups");

    for i in 0..average_speedups.len() {
        println!("Num generations: {}", num_gen_options[i]);
        println!("Average speedup: {}", average_speedups[i]);
        println!("Average fitness: {}", average_fitnesses[i]);
        println!("Average permanence: {}", average_permanences[i]);
    }
    println!("Immediate successor speedup: {}", last_immediate_successor_speedup);

    let start = Instant::now();
    //gen_scatter_evolution(TEMP_FILE_NAME, "test");
    println!("Time to generate surprise evolution graph: {:?}", start.elapsed());
}

#[allow(dead_code)]
fn test_metaheuristics_04(num_iter: usize) {

    let mut report = Vec::with_capacity(20);
    let start = Instant::now();

    let num_nodes = 64;
    let num_edges = 64;
    let min_parallelism = 16;
    let mixing_coeff = 0.1;
    let num_communities = 4;
    let max_comm_size_difference = 0;

    //let g = gen_random_digraph(true, 16, Some(160), 32, Some(600), Some(32));
    //let g = gen_random_digraph(true, 16, Some(num_nodes), 32, Some(num_edges), Some(min_parallelism));
    let g = gen_lfr_like_graph(num_nodes, num_edges, mixing_coeff, num_communities, max_comm_size_difference);
    println!("Time to generate random graph: {:?}", start.elapsed());

    const TEMP_FILE_NAME: &str = "metaheuristics_evolution.csv";

    let mut _f = File::create(TEMP_FILE_NAME).unwrap();
    const POP_SIZE: usize = 64;

    let mut best_surprise_per_algo = HashMap::new();

    let mut average_speedups: Vec<f64> = vec![];
    let mut average_fitnesses: Vec<f64> = vec![];
    let mut average_permanences: Vec<f64> = vec![];

    let num_gen_options = [0, 100, 400, 8000];
    for num_generations in num_gen_options {
        let mut speedup_sum = 0.0;
        let mut fitness_sum = 0.0;
        let mut permanence_sum = 0.0;

        for _ in 0..num_iter {
            /*
            let start = Instant::now();

            //run_algo!(Fa::default(), report, g);
            let _s = Solver::build(Fa::default(), BaseSolver{graph: &g})
                .task(|ctx| ctx.gen == num_generations)
                .pop_num(POP_SIZE)
                .callback(|ctx| report.push(ctx.best_f))
                .solve()
                .unwrap();

            let duration = start.elapsed();
            println!("Elapsed time (Firefly algorithm): {:?}", duration);
            println!("Time per iteration: {:?}", duration / (num_generations as u32));

            //write_report_and_clear("Firefly", &mut report, &mut _f);

            let mut algo_best = *best_surprise_per_algo.entry("Firefly").or_insert(f64::MAX);

            if _s.best_fitness() < algo_best {
                algo_best = _s.best_fitness();
                visualize_graph(&g, Some(&_s.result()), Some(format!("firefly_{}_{}_{}", POP_SIZE, num_generations, algo_best)));
            }
            */

            /*
            let test_finalized_inner = [0; MAX_NUMBER_OF_NODES - 50];
            let test_finalized: Option<&[usize]> = Some(&test_finalized_inner);
            let _s = de_solve(&g, num_generations, POP_SIZE, Some(&mut report), test_finalized);
            */

            //let mut mut_core_bound = vec![[2., 2.]; MAX_NUMBER_OF_NODES];
            //mut_core_bound[5] = [3., 3.];
            //let core_bound = &mut_core_bound;

            //let core_bound = &vec![[0., MAX_NUMBER_OF_PARTITIONS as f64]; MAX_NUMBER_OF_NODES];
            let core_bound = &build_bound_vec_for_compact_solver();
            let _s = de_compact_solve(&g, num_generations, POP_SIZE, Some(&mut report), None, true, core_bound);

            let start = Instant::now();
            //visualize_graph(&g, Some(&_s.result()), Some(format!("differential_evolution_{}_{}_{}", POP_SIZE, num_generations, algo_best)));
            println!("Time to generate graph visualization: {:?}", start.elapsed());
            let start = Instant::now();
            let mut pid_array = [0; MAX_NUMBER_OF_NODES];
            expand_compact_pid(&_s.result(), &mut pid_array);
            let (finalized_core_placements, execution_info) = evaluate_execution_time_and_speedup(&g, &pid_array, num_generations, false);
            //let (finalized_core_placements, execution_info) = evaluate_execution_time_and_speedup(&g, &_s.result(), num_generations, false);
            println!("Time to simulate execution: {:?}", start.elapsed());
            speedup_sum += execution_info.speedup;
            fitness_sum += _s.best_fitness();
            let start = Instant::now();
            permanence_sum += calculate_permanence(&g, &finalized_core_placements, &_s.result());
            println!("Time to calculate permanence: {:?}", start.elapsed());
            println!("{:?}", execution_info);
            println!("Single-shot surprise: {:?}", _s.best_fitness());
            println!("_s.result():\t\t\t{:?}", _s.result());
            println!("finalized_core_placements:\t{:?}", finalized_core_placements);

            let algo_best = *best_surprise_per_algo.entry("Differential Evolution").or_insert(f64::MIN);
            //if _s.best_fitness() < algo_best {
            if execution_info.speedup > algo_best {
                //best_surprise_per_algo.insert("Differential Evolution", _s.best_fitness());
                best_surprise_per_algo.insert("Differential Evolution", execution_info.speedup);
                let mut res: Vec<Option<usize>> = vec![];
                _s.result().into_iter().for_each(|v| res.push(Some(v)));

                visualize_graph(&g, Some(&res), Some(format!("differential_evolution_predicted_placement_{}_{}_{}", POP_SIZE, num_generations, _s.best_fitness())));
                visualize_graph(&g, Some(&finalized_core_placements), Some(format!("differential_evolution_final_placement_{}_{}_{}_{}", POP_SIZE, num_generations, _s.best_fitness(), execution_info.speedup)));
                println!("Exact speedup: {:.32}", execution_info.speedup);
            }

            /*
            let _s = Solver::build(Pso::default(), BaseSolver{graph: &g})
                .task(|ctx| ctx.gen == num_generations)
                .pop_num(POP_SIZE)
                .callback(|ctx| report.push(ctx.best_f))
                .solve()
                .unwrap();

            write_report_and_clear("Particle Swarm Optimization", &mut report, &mut _f);

            let _s = Solver::build(Rga::default(), BaseSolver{graph: &g})
                .task(|ctx| ctx.gen == num_generations)
                .pop_num(POP_SIZE)
                .callback(|ctx| report.push(ctx.best_f))
                .solve()
                .unwrap();

            write_report_and_clear("Real-Coded Genetic Algorithm", &mut report, &mut _f);

            let _s = Solver::build(Tlbo::default(), BaseSolver{graph: &g})
                .task(|ctx| ctx.gen == num_generations)
                .pop_num(POP_SIZE)
                .callback(|ctx| report.push(ctx.best_f))
                .solve()
                .unwrap();

            write_report_and_clear("Teaching Learning Based Optimization", &mut report, &mut _f);
            */
        }
        let average_speedup = speedup_sum / (num_iter as f64);
        let average_fitness = fitness_sum / (num_iter as f64);
        let average_permanence = permanence_sum / (num_iter as f64);

        average_speedups.push(average_speedup);
        average_fitnesses.push(average_fitness);
        average_permanences.push(average_permanence);
        _f.write(format!("Differential Evolution,{},{},fitness_test,{},{},{},{},{},{}\n", num_generations, average_fitness, average_speedup, MAX_NUMBER_OF_PARTITIONS, num_nodes, num_edges, min_parallelism, average_permanence).as_bytes()).unwrap();
    }

    gen_speedup_bars(TEMP_FILE_NAME, "speedups");

    for i in 0..average_speedups.len() {
        println!("Num generations: {}", num_gen_options[i]);
        println!("Average speedup: {}", average_speedups[i]);
        println!("Average fitness: {}", average_fitnesses[i]);
        println!("Average permanence: {}", average_permanences[i]);
    }

    let start = Instant::now();
    //gen_scatter_evolution(TEMP_FILE_NAME, "test");
    println!("Time to generate surprise evolution graph: {:?}", start.elapsed());
}

fn main() {
    //gen_sample_graph_image();
    //test_histogram_01();
    //test_metaheuristics_01();
    //test_metaheuristics_02();
    test_metaheuristics_03(10);
    
    /*
    for i in [1,2,3] {
        println!("{i}");
    }
    */
}
