use petgraph::graph::*;
use petgraph::{Directed, Undirected};
use petgraph::EdgeType;
use petgraph::graphmap::GraphMap;
use petgraph::graphmap::*;
use petgraph::visit::{IntoNodeReferences, IntoEdgeReferences};
use petgraph::visit::NodeRef;
use petgraph::algo::{dijkstra, min_spanning_tree};
use petgraph::data::FromElements;
use petgraph::dot::{Dot, Config};
use petgraph::visit::EdgeRef;
use std::fs::File;
use std::io::prelude::*;
use std::fmt;
use std::process::Command;
use rand::Rng;
use rand::seq::SliceRandom;
use std::cmp::min;
use statrs::function::factorial::binomial;
use std::collections::{HashMap, HashSet, BTreeMap};
use csv::Writer;
use std::time::{Duration, Instant};
use metaheuristics_nature::ndarray::{Array2, ArrayBase, OwnedRepr, Dim};
use rustc_hash::{FxHashMap, FxHashSet};

use metaheuristics_nature::{Rga, Fa, Pso, De, Tlbo, Solver};
use metaheuristics_nature::tests::TestObj;

use once_cell::sync::Lazy;
use dinic_maxflow::*;

use scan_fmt::*;
mod dinic_maxflow;

const PRINT_WEIGHTS: bool = false;
const MAX_NUMBER_OF_PARTITIONS: usize = 40;
const KERNEL_NUMBER_OF_PARTITIONS: usize = 2;
const MAX_NUMBER_OF_NODES: usize = 257;
//const MAX_BINOMIAL_A: usize = 550000;
const MAX_BINOMIAL_A: usize = 33000;
const MAX_BINOMIAL_B: usize = 1024;
/*
const MAX_BINOMIAL_A: usize = 0;
const MAX_BINOMIAL_B: usize = 0;
*/

static BIN_MATRIX: Lazy<Vec<Vec<f64>>> = Lazy::new(|| {
    let m = new_binomial_matrix(MAX_BINOMIAL_A, MAX_BINOMIAL_B);
    println!("Finished initializing binomial matrix.");
    return m;
});

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

impl NodeInfo {
    pub fn new(n_id: usize) -> Self {
        Self {
            numerical_id: n_id,
            partition_id: 0
        }
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
    fn internal_edge_count_and_max_internal_links(&self, pid_array: Option<&[usize]>) -> (u64, u64, u64);
    fn calculate_max_internal_links(&self, pid_array: Option<&[usize]>) -> usize;
}

impl CanCountInternalLinks for Graph<NodeInfo, usize, Directed, usize> {
    fn internal_edge_count_and_max_internal_links(&self, pid_array: Option<&[usize]>) -> (u64, u64, u64) {
        let mut num_internal_links : usize = 0;
        let mut items_per_partition = HashMap::new();
        let mut max_internal_links = 0;
        let mut num_links = 0;

        if pid_array.is_none() {
            for v in self.node_references() {
                let pid = v.weight().partition_id();
                //println!("[internal_edge_count_and_max_internal_links]: Visiting node {:?}, belonging to partition {}", v, pid);

                let hash_count = items_per_partition.entry(pid).or_insert(0);
                *hash_count += 1;

                for n in self.neighbors(v.id()) {
                    //println!("[internal_edge_count_and_max_internal_links]: Visiting neighbor {:?}", n);
                    let neighbor_weight = self.node_weight(n).unwrap();
                    let npid = neighbor_weight.partition_id;
                    
                    let edge = self.edges_connecting(v.id(), n.id()).last().unwrap();
                    let edge_weight = edge.weight();
                    if  npid == pid {
                        num_internal_links += *edge_weight;
                    }

                    num_links += *edge_weight;
                }
            }
        } else {
            for v in self.node_references() {
                let pid_array = pid_array.unwrap();

                let pid = pid_array[v.weight().numerical_id];

                let hash_count = items_per_partition.entry(pid).or_insert(0);
                *hash_count += 1;

                //let pid = pid_array[v.id().index()];
                //println!("[internal_edge_count_and_max_internal_links]: Visiting node {:?}, belonging to partition {}", v, pid);

                for n in self.neighbors(v.id()) {
                    //println!("[internal_edge_count_and_max_internal_links]: Visiting neighbor {:?}", n);
                    let neighbor_weight = self.node_weight(n).unwrap();
                    let npid = pid_array[neighbor_weight.numerical_id];
                    
                    let edge = self.edges_connecting(v.id(), n.id()).last().unwrap();
                    let edge_weight = edge.weight();
                    if  npid == pid {
                        num_internal_links += *edge_weight;
                    }

                    num_links += *edge_weight;
                }
            }
        }

        for n in items_per_partition.values() {
            max_internal_links += n * (n - 1) / 2;
        }

        //[println!("[internal_edge_count_and_max_internal_links]: {}, {}, {}", num_internal_links, max_internal_links, num_links);
        (num_internal_links as u64, max_internal_links, num_links as u64)
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

fn new_binomial_matrix(max_a: usize, max_b: usize) -> Vec<Vec<f64>> {
    let mut v: Vec<Vec<f64>> = vec![vec![]; max_a + 1];

    for i in 0..=max_a {
        for j in 0..=max_b {
            let mut partial = 0.0;

            if j <= i {
                if i > (MAX_BINOMIAL_B) && i <= (MAX_BINOMIAL_A) && j <= (MAX_BINOMIAL_B) {
                    partial = v[i - 1][j];
                    partial += (i as f64).log2();
                    partial -= ((i - j) as f64).log2();
                } else {
                    for k in 0..(i - j) {
                        partial += ((j + 1 + k) as f64).log2();
                        partial -= ((1 + k) as f64).log2();
                    }
                }
            }

            //println!("[new_binomial]: (i, j): ({}, {}) = {}", i, j, partial);

            v[i].push(partial);
        }
    }

    v
}

fn new_binomial(a_raw: u64, b_raw: u64, force_compute: bool) -> f64 {
    // a_raw: large
    // b_raw: small
    
    if !force_compute && a_raw <= (MAX_BINOMIAL_A as u64) && b_raw <= (MAX_BINOMIAL_B as u64) {
        //println!("[new_binomial]: (a_raw, b_raw) = ({}, {})", a_raw, b_raw);
        return BIN_MATRIX[a_raw as usize][b_raw as usize];
    } else {
        let mut partial = 0.0;

        for i in 0..(a_raw - b_raw) {
            partial += ((b_raw + 1 + i) as f64).log2();
            partial -= ((1 + i) as f64).log2();
        }

        println!("[new_binomial]: (a_raw, b_raw): ({}, {}) = {}", a_raw, b_raw, partial);

        return partial;
    }
}

// Implemented according to the definition found in https://doi.org/10.1371/journal.pone.0024195 .
// Better clusterings have a higher surprise value.
#[allow(dead_code)]
fn original_calculate_surprise(g: &Graph<NodeInfo, usize, Directed, usize>, pid_array: Option<&[usize]>) -> f64 {
    let num_nodes = g.node_count();

    //let num_links : u64 = g.edge_count() as u64;
    let num_max_links : u64 = (num_nodes * (num_nodes - 1) / 2) as u64;

    let (num_internal_links, num_max_internal_links, num_links) : (u64, u64, u64) = g.internal_edge_count_and_max_internal_links(pid_array);
    //let num_max_internal_links = g.calculate_max_internal_links(pid_array) as u64;

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
    let mut partition_sizes = [0; KERNEL_NUMBER_OF_PARTITIONS];
    let num_partition_elements = KERNEL_NUMBER_OF_PARTITIONS - 1;

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

        for step in 0..KERNEL_NUMBER_OF_PARTITIONS - 1 {
            let target_pid = (step + base_pid_step) % KERNEL_NUMBER_OF_PARTITIONS;
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
fn calculate_surprise(g: &Graph<NodeInfo, usize, Directed, usize>, pid_array: Option<&[usize]>) -> f64 {
    let num_nodes = g.node_count();

    //let num_links : u64 = g.edge_count() as u64;
    let num_max_links : u64 = (num_nodes * (num_nodes - 1) / 2) as u64;

    let (num_internal_links, num_max_internal_links, num_links) : (u64, u64, u64) = g.internal_edge_count_and_max_internal_links(pid_array);
    //let num_max_internal_links = g.calculate_max_internal_links(pid_array) as u64;

    let mut surprise: f64 = 0.0;

    let top = min(num_links, num_max_internal_links);

    let j = (num_internal_links + top) / 2;
    surprise += new_binomial(num_max_internal_links, j, false) + new_binomial(num_max_links - num_max_internal_links, num_links - j, false) - (num_internal_links as f64) * 0.001;
    //surprise += new_binomial(num_max_internal_links, j, false) + new_binomial(num_max_links - num_max_internal_links, num_links - j, false) - (num_internal_links as f64) * 0.01;
    //surprise += new_binomial(num_max_internal_links, j, false) + new_binomial(num_max_links - num_max_internal_links, num_links - j, false);

    //surprise -= new_binomial(num_max_links, num_links);

    //println!("Graph surprise: {}", surprise);
    
    //surprise = surprise + calculate_pid_array_min_max_distance(pid_array, num_nodes) * 10.;

    surprise 
}

// Implemented according to Chakraborty, Tanmoy, et al. "On the permanence of vertices in network communities."
// Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014.
// https://doi.org/10.1145/2623330.2623707
fn node_permanence(nid: Option<usize>, v: Option<NodeIndex<usize>>, original_graph: &Graph<NodeInfo, usize, Directed, usize>, finalized_core_placements: &[Option<usize>], pid_array: &[usize]) -> f64 {
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
    //for n in original_graph.neighbors_undirected(v) {
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
            //for nk in original_graph.neighbors_undirected(k.id()) {
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

fn calculate_permanence(original_graph: &Graph<NodeInfo, usize, Directed, usize>, finalized_core_placements: &[Option<usize>], pid_array: &[usize]) -> f64 {
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

fn gen_random_digraph(acyclic: bool, max_num_nodes: usize, exact_num_nodes: Option<usize>, max_num_edges: usize, exact_num_edges: Option<usize>, min_parallelism: Option<usize>) -> Graph<NodeInfo, usize, Directed, usize> {
    let mut g = GraphMap::<NodeInfo, usize, Directed>::new();
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
    
    let g : Graph<NodeInfo, usize, Directed, usize> = g.into_graph();

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

fn set_random_partitions(g: &mut Graph<NodeInfo, usize, Directed, usize>, max_partitions: usize) {
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
fn get_number_of_partitions(g: &Graph<NodeInfo, usize, Directed, usize>) -> usize {
    let mut items_per_partition = HashMap::new();

    for v in g.node_references() {
        let pid = v.weight().partition_id();
        let _ = items_per_partition.entry(pid).or_insert(0);
    }

    items_per_partition.len()
}

fn visualize_graph<T: EdgeType>(g: &Graph<NodeInfo, usize, T, usize>, pid_array: Option<&[Option<usize>]>, output_name: Option<String>) {
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

    fn weight_label<T: EdgeType> (_: &Graph<NodeInfo, usize, T, usize>, edge_ref: EdgeReference<'_, usize, usize>) -> String {
        format!("label={:?}", edge_ref.weight()).to_string()
    }

    fn node_attr_generator<P: petgraph::visit::NodeRef, T: EdgeType>(_: &Graph<NodeInfo, usize, T, usize>, node_ref: P) -> String where <P as petgraph::visit::NodeRef>::Weight: fmt::Debug + HasPartition {
        //let new_node_ref: NodeIndex;
        //new_node_ref = node_ref.into();

        let w = node_ref.weight();
        //let c = COLORS[w.partition_id()];
        let c = get_equally_hue_spaced_hsv_string(w.partition_id(), MAX_NUMBER_OF_PARTITIONS);
        format!("style=filled, color=\"{}\", fillcolor=\"{}\"", c, c).to_string()
    }

    let dot_dump = if PRINT_WEIGHTS {
        format!("{:?}", Dot::with_attr_getters(&g, &[Config::EdgeNoLabel], &weight_label, &node_attr_generator))
    } else {
        let null_out = |_, _| "".to_string();
        format!("{:?}", Dot::with_attr_getters(&g, &[Config::EdgeNoLabel], &null_out, &node_attr_generator))
    };

    let _ = write_to_file(&dot_dump, out_name_unwrapped);
    //calculate_surprise(&g);
}

fn set_random_partitions_and_visualize_graph(graph: &mut Graph<NodeInfo, usize, Directed, usize>, max_partitions: usize) -> &Graph<NodeInfo, usize, Directed, usize>{
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
fn evaluate_multiple_random_clusterings(original_graph: &Graph<NodeInfo, usize, Directed, usize>, max_partitions: usize, num_iterations: usize, gen_image: bool) {
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
    let new_graph = set_random_partitions_and_visualize_graph(&mut g, KERNEL_NUMBER_OF_PARTITIONS);
    visualize_graph(&new_graph, None, None);
}

/*
#[allow(dead_code)]
fn test_histogram_01() {
    let g = gen_random_digraph(true, 16, Some(16), 0, None, None);
    evaluate_multiple_random_clusterings(&g, KERNEL_NUMBER_OF_PARTITIONS, 100000000, true);
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
    graph: &'a Graph<NodeInfo, usize, Directed, usize>,
    core_bound: &'a [[f64; 2]]
}

impl Bounded for BaseSolver<'_> {
    fn bound(&self) -> &[[f64; 2]] {
        self.core_bound
        //&[[0., KERNEL_NUMBER_OF_PARTITIONS as f64]; MAX_NUMBER_OF_NODES]
    }
}

struct CompactSolver<'a> {
    graph: &'a Graph<NodeInfo, usize, Directed, usize>,
    core_bound: &'a [[f64; 2]]
}

impl Bounded for CompactSolver<'_> {
    fn bound(&self) -> &[[f64; 2]] {
        self.core_bound
        //&[[0., KERNEL_NUMBER_OF_PARTITIONS as f64]; MAX_NUMBER_OF_NODES]
    }
}

fn round_float_array(float_arr: &[f64]) -> [usize; MAX_NUMBER_OF_NODES] {
    let mut int_arr: [usize; MAX_NUMBER_OF_NODES] = [0; MAX_NUMBER_OF_NODES];

    for i in 0..float_arr.len() {
        int_arr[i] = float_arr[i].round() as usize;
    }

    int_arr
}

fn floor_float_array(float_arr: &[f64]) -> Vec<usize> {
    //let mut int_arr: [usize; MAX_NUMBER_OF_NODES] = [0; MAX_NUMBER_OF_NODES];
    let mut int_arr: Vec<usize> = vec![];

    for f in float_arr {
        int_arr.push(*f as usize % KERNEL_NUMBER_OF_PARTITIONS);
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
        //400. + (-original_calculate_surprise(&self.graph, Some(&x))).log2()
        //1000.0 + calculate_surprise(&self.graph, Some(&x))
        
        calculate_surprise(&self.graph, Some(&x))
        //1. - calculate_permanence(&self.graph, &[None; MAX_NUMBER_OF_NODES], &x)
    }
}

impl ObjFactory for CompactSolver<'_> {
    //type Product = [usize; MAX_NUMBER_OF_NODES];
    type Product = [usize; KERNEL_NUMBER_OF_PARTITIONS - 1 + MAX_NUMBER_OF_NODES];
    type Eval = f64;

    fn produce(&self, xs: &[f64]) -> Self::Product {
        const C_SIZE: usize = KERNEL_NUMBER_OF_PARTITIONS - 1 + MAX_NUMBER_OF_NODES;
        let mut compact_pid_array = [0; C_SIZE];
        
        for i in 0..C_SIZE {
            compact_pid_array[i] = xs[i] as usize;
        }

        compact_pid_array
    }

    //fn evaluate(&self, x: [usize; MAX_NUMBER_OF_NODES]) -> Self::Eval {
    fn evaluate(&self, x: [usize; KERNEL_NUMBER_OF_PARTITIONS - 1 + MAX_NUMBER_OF_NODES]) -> Self::Eval {
        //400. + (-original_calculate_surprise(&self.graph, Some(&x))).log2()
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

fn get_num_pending_deps(graph: &Graph<NodeInfo, usize, Directed, usize>) -> Vec<usize>{
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
        g: &mut Graph<NodeInfo, usize, Directed, usize>,
        original_graph: &Graph<NodeInfo, usize, Directed, usize>,
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
fn retire_finished_tasks(g: &mut Graph<NodeInfo, usize, Directed, usize>, core_states: &mut Vec<Option<ExecutionUnit>>, pid_array_option: Option<&mut [usize]>) -> Option<usize> {
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

fn num_foreign_incoming_edges(nid: usize, original_graph: &Graph<NodeInfo, usize, Directed, usize>, target_core: usize, finalized_core_placements: &mut [Option<usize>]) -> usize {
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

fn num_native_edges(nid: usize, original_graph: &Graph<NodeInfo, usize, Directed, usize>, finalized_core_placements: &mut [Option<usize>]) -> usize {
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

fn num_edges_to_target_core(nid: usize, original_graph: &Graph<NodeInfo, usize, Directed, usize>, target_core: usize, finalized_core_placements: &mut [Option<usize>]) -> usize {
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

fn get_task_with_lowest_cluster_degree(ready_queues: &mut Vec<Vec<ExecutionUnit>>, original_graph: &Graph<NodeInfo, usize, Directed, usize>, finalized_core_placements: &mut [Option<usize>]) -> Option<ExecutionUnit> {
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

fn get_task_with_lowest_extra_expected_misses(ready_queues: &mut Vec<Vec<ExecutionUnit>>, original_graph: &Graph<NodeInfo, usize, Directed, usize>, finalized_core_placements: &mut [Option<usize>], target_core: usize) -> Option<ExecutionUnit> {
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

fn feed_idle_cores(ready_queues: &mut Vec<Vec<ExecutionUnit>>, core_states: &mut Vec<Option<ExecutionUnit>>, original_graph: &Graph<NodeInfo, usize, Directed, usize>, finalized_core_placements: &mut [Option<usize>]) -> (usize, usize) {
    let mut num_misses = 0;
    let mut num_tasks_stolen = 0;
    let mut core_permutation: [usize; MAX_NUMBER_OF_PARTITIONS] = core::array::from_fn(|i| i);
    let mut rng = rand::thread_rng();

    core_permutation.shuffle(&mut rng);

    for steal in [false, true] {
    //for steal in [false] {
        //for i in 0..MAX_NUMBER_OF_PARTITIONS {
        for i in core_permutation {
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

fn evaluate_execution_time_and_speedup(original_graph: &Graph<NodeInfo, usize, Directed, usize>, pid_array: &[usize], num_generations: usize, immediate_successor: bool) -> ([Option<usize>; MAX_NUMBER_OF_NODES], ExecutionInfo) {
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
    probe_step_size: usize,
    probe_step_vec: Option<Vec<usize>>
}

fn de_solve<'a>(g: &'a Graph<NodeInfo, usize, Directed, usize>, num_generations: usize, population_size: usize, report: Option<&mut Vec<f64>>, finalized_core_placements: Option<&'a [Option<usize>]>, verbose: bool, core_bound: &'a [[f64; 2]], config: &RunConfig, partial_solutions: &mut Vec<(usize, Vec<usize>)>) -> Solver<BaseSolver<'a>>{
    let start = Instant::now();

    //let core_bound = &[[0., KERNEL_NUMBER_OF_PARTITIONS as f64]; MAX_NUMBER_OF_NODES][..];

    if report.is_some() {
        let report = report.unwrap();

        //let _s = Solver::build(De::default().f(0.5), BaseSolver{graph: &g, core_bound})
        let _s = Solver::build(De::default(), BaseSolver{graph: &g, core_bound})
        //let _s = Solver::build(Fa::default().alpha(0.5), BaseSolver{graph: &g, core_bound})
            .task(|ctx| ctx.gen == num_generations as u64)
            .pop_num(population_size)
            //.pool(|ctx, rng| Array2::from_shape_fn(ctx.pool_size(), |(_, s)| rng.range(ctx.func.bound_range(s))))
            .pool(|ctx, rng| Array2::from_shape_fn(ctx.pool_size(), |(i, j)| {
                if i > 1 {
                    return rng.range(ctx.func.bound_range(j));
                } else {
                    return 0.;
                }
                /*
                } else {
                    if i == 0 {
                        return 0.;
                    } else {
                        return 0.;
                    }
                }
                */
            }))
            .callback(|ctx| {
                if config.probe_step_vec.is_none() {
                    if ctx.gen % config.probe_step_size as u64 == 0 {
                        report.push(ctx.best_f);
                        partial_solutions.push((ctx.gen as usize, ctx.result()));
                    }
                } else {
                    if config.probe_step_vec.clone().unwrap().contains(&(ctx.gen as usize)) {
                        //println!("pushed at gen = {}", ctx.gen);
                        //println!("linspace_vec: {:?}", config.probe_step_vec.clone().unwrap());
                        report.push(ctx.best_f);
                        partial_solutions.push((ctx.gen as usize, ctx.result()));
                    }
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
                if i > 1 {
                    return rng.range(ctx.func.bound_range(j));
                } else {
                    return 0.;
                }
                /*
                } else {
                    if i == 0 {
                        return 0.;
                    } else {
                        return 0.;
                    }
                }
                */
            }))
            .callback(|ctx| {
                if config.probe_step_vec.is_none() {
                    if ctx.gen % config.probe_step_size as u64 == 0 {
                        partial_solutions.push((ctx.gen as usize, ctx.result()));
                    }
                } else {
                    if config.probe_step_vec.clone().unwrap().contains(&(ctx.gen as usize)) {
                        //println!("pushed at gen = {}", ctx.gen);
                        //println!("linspace_vec: {:?}", config.probe_step_vec.clone().unwrap());
                        partial_solutions.push((ctx.gen as usize, ctx.result()));
                    }
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

        return _s;
    }
}

fn de_compact_solve<'a>(g: &'a Graph<NodeInfo, usize, Directed, usize>, num_generations: usize, population_size: usize, report: Option<&mut Vec<f64>>, finalized_core_placements: Option<&'a [Option<usize>]>, verbose: bool, core_bound: &'a [[f64; 2]]) -> Solver<CompactSolver<'a>>{
    let start = Instant::now();

    //let core_bound = &[[0., KERNEL_NUMBER_OF_PARTITIONS as f64]; MAX_NUMBER_OF_NODES][..];

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

fn gen_lfr_like_graph<T: EdgeType>(num_nodes: usize, num_edges: usize, mixing_coeff: f64, num_communities: usize, max_comm_size_difference: usize) -> Graph<NodeInfo, usize, T, usize> {
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

    let mut g = GraphMap::<NodeInfo, usize, T>::with_capacity(num_nodes, num_edges);
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

    let g: Graph<NodeInfo, usize, T, usize> = g.into_graph();
    g
}

fn graph_splitter(g: &Graph<NodeInfo, usize, Directed, usize>, pid_array: &[usize]) -> Vec<Graph<NodeInfo, usize, Directed, usize>>{
    // TODO: Extend this for graphs with more than two clusters
    //let start = Instant::now();
    let mut g0 = g.clone();
    let mut g1 = g.clone();
    //let duration = start.elapsed();
    //println!("Time for cloning graphs: {:?}", duration);

    let mut graphs_vec: Vec<Graph<NodeInfo, usize, Directed, usize>> = vec![];

    for n in g.node_references() {
        let n_nid = g.node_weight(n.id()).unwrap().numerical_id;
        let n_pid = pid_array[n_nid];

        if n_pid == 0 {
            let to_remove = g1.node_references().find(|x| x.1.numerical_id == n_nid).unwrap().id();
            g1.remove_node(to_remove);
        } else {
            let to_remove = g0.node_references().find(|x| x.1.numerical_id == n_nid).unwrap().id();
            g0.remove_node(to_remove);
        }
    }

    graphs_vec.push(g0);
    graphs_vec.push(g1);

    graphs_vec
}

fn array_to_vec(pid_array: &[usize]) -> Vec<Option<usize>> {
    let mut res: Vec<Option<usize>> = vec![];
    pid_array.into_iter().for_each(|v| res.push(Some(*v)));
    res
}

#[allow(dead_code)]
fn test_metaheuristics_03(num_iter: usize) {

    let mut report = Vec::with_capacity(20);
    let start = Instant::now();

    let num_nodes = 128;
    let num_edges = 64;
    let min_parallelism = 16;
    let mixing_coeff = 0.003;
    let num_communities = 32;
    let max_comm_size_difference = 0;

    //let g = gen_random_digraph(true, 16, Some(160), 32, Some(600), Some(32));
    //let g = gen_random_digraph(true, 16, Some(num_nodes), 32, Some(num_edges), Some(min_parallelism));
    let g = gen_lfr_like_graph(num_nodes, num_edges, mixing_coeff, num_communities, max_comm_size_difference);
    println!("Time to generate random graph: {:?}", start.elapsed());

    const TEMP_FILE_NAME: &str = "metaheuristics_evolution.csv";
    const NUM_EVALUATIONS: usize = 50;

    let mut _f = File::create(TEMP_FILE_NAME).unwrap();
    const POP_SIZE: usize = 64;

    let mut best_surprise_per_algo = HashMap::new();

    let mut average_speedups: Vec<f64> = vec![];
    let mut average_fitnesses: Vec<f64> = vec![];
    let mut average_permanences: Vec<f64> = vec![];
    let mut immediate_successor_average: f64 = 0.;

    let num_gen_options = [4000];
    //let num_gen_options = [0];
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

            let core_bound = &vec![[0., KERNEL_NUMBER_OF_PARTITIONS as f64]; MAX_NUMBER_OF_NODES];
            let conf = RunConfig{probe_step_size: 100, probe_step_vec: None};
            let mut partial_solutions: Vec<(usize, Vec<usize>)> = vec![];
            let _s = de_solve(&g, num_generations, POP_SIZE, Some(&mut report), None, true, core_bound, &conf, &mut partial_solutions);

            let mut partial_speedups: Vec<(usize, f64)> = vec![];
            let mut calc_time = Duration::new(0, 0);
            for i in 0..partial_solutions.len() {
                let partial_solution = &partial_solutions[i];
                let pid_array = &partial_solution.1;
                let mut speedup: f64 = 0.;
                let mut permanence: f64 = 0.;

                let start = Instant::now();
                for _ in 0..NUM_EVALUATIONS {
                    let (finalized_core_placements, execution_info) = evaluate_execution_time_and_speedup(&g, pid_array, 0, false);
                    let void_finalized = [None; MAX_NUMBER_OF_NODES];
                    permanence += calculate_permanence(&g, &void_finalized, pid_array);
                    speedup += execution_info.speedup;
                }
                calc_time += start.elapsed();


                permanence /= NUM_EVALUATIONS as f64;
                speedup /= NUM_EVALUATIONS as f64;

                let num_gen = partial_solution.0;
                let fitness = report[i];

                //let permanence = calculate_permanence(&g, &finalized_core_placements, pid_array);

                partial_speedups.push((num_gen, speedup));
                _f.write(format!("Differential Evolution,{},{},fitness_test,{},{},{},{},{},{}\n", num_gen, fitness, speedup, KERNEL_NUMBER_OF_PARTITIONS, num_nodes, num_edges, min_parallelism, permanence).as_bytes()).unwrap();
            }
            println!("Time to evaluate executions {} times and calculate speedup and permanence: {:?}", NUM_EVALUATIONS * partial_solutions.len(), calc_time);

            println!("{:?}", partial_speedups);

            for _ in 0..NUM_EVALUATIONS {
                let (immediate_successor_finalized_placements, immediate_successor_execution_info) = evaluate_execution_time_and_speedup(&g, &_s.result(), num_generations, true);
                immediate_successor_average += immediate_successor_execution_info.speedup / (NUM_EVALUATIONS * num_iter) as f64;

                let algo_best = *best_surprise_per_algo.entry("Random Immediate Successor").or_insert(f64::MIN);
                if immediate_successor_execution_info.speedup > algo_best {
                    best_surprise_per_algo.insert("Random Immediate Successor", immediate_successor_execution_info.speedup);
                    visualize_graph(&g, Some(&immediate_successor_finalized_placements), Some(format!("random_immediate_successor_{}", immediate_successor_execution_info.speedup)));
                }
            }

            let mut first = true;
            for _ in 0..NUM_EVALUATIONS {
                let start = Instant::now();
                let (finalized_core_placements, execution_info) = evaluate_execution_time_and_speedup(&g, &_s.result(), num_generations, false);

                if first {
                    println!("Time to simulate execution: {:?}", start.elapsed());
                }

                speedup_sum += execution_info.speedup;
                fitness_sum += _s.best_fitness();
                permanence_sum += calculate_permanence(&g, &finalized_core_placements, &_s.result());

                if first {
                    first = false;
                    println!("{:?}", execution_info);
                    println!("Single-shot surprise: {:?}", _s.best_fitness());
                    println!("_s.result():\t\t\t{:?}", _s.result());
                    println!("finalized_core_placements:\t{:?}", finalized_core_placements);
                }

                let algo_best = *best_surprise_per_algo.entry("Differential Evolution").or_insert(f64::MIN);
                if execution_info.speedup > algo_best {
                    //best_surprise_per_algo.insert("Differential Evolution", _s.best_fitness());
                    best_surprise_per_algo.insert("Differential Evolution", execution_info.speedup);
                    let mut res: Vec<Option<usize>> = vec![];
                    _s.result().into_iter().for_each(|v| res.push(Some(v)));

                    visualize_graph(&g, Some(&res), Some(format!("differential_evolution_predicted_placement_{}_{}_{}", POP_SIZE, num_generations, _s.best_fitness())));
                    visualize_graph(&g, Some(&finalized_core_placements), Some(format!("differential_evolution_final_placement_{}_{}_{}_{}", POP_SIZE, num_generations, _s.best_fitness(), execution_info.speedup)));
                    println!("Exact speedup: {:.32}", execution_info.speedup);
                }
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
        let average_speedup = speedup_sum / ((NUM_EVALUATIONS * num_iter) as f64);
        let average_fitness = fitness_sum / ((NUM_EVALUATIONS * num_iter) as f64);
        let average_permanence = permanence_sum / ((NUM_EVALUATIONS * num_iter) as f64);

        average_speedups.push(average_speedup);
        average_fitnesses.push(average_fitness);
        average_permanences.push(average_permanence);
        //_f.write(format!("Differential Evolution,{},{},fitness_test,{},{},{},{},{},{}\n", num_generations, average_fitness, average_speedup, KERNEL_NUMBER_OF_PARTITIONS, num_nodes, num_edges, min_parallelism, average_permanence).as_bytes()).unwrap();
    }

    gen_speedup_bars(TEMP_FILE_NAME, "speedups");

    for i in 0..average_speedups.len() {
        println!("Num generations: {}", num_gen_options[i]);
        println!("Average speedup: {}", average_speedups[i]);
        println!("Average fitness: {}", average_fitnesses[i]);
        println!("Average permanence: {}", average_permanences[i]);
    }
    println!("Immediate successor speedup: {}", immediate_successor_average);

    let start = Instant::now();
    //gen_scatter_evolution(TEMP_FILE_NAME, "test");
    println!("Time to generate surprise evolution graph: {:?}", start.elapsed());
}

fn merge_pid_arrays_from_solutions(solutions: &Vec<Solver<BaseSolver<'_>>>, graphs: &Vec<Graph<NodeInfo, usize, Directed, usize>>, num_nodes: usize) -> Vec<Option<usize>>{
    let mut merged_pid_array: Vec<Option<usize>> = vec![None; num_nodes];
    let pid_arrays: Vec<Vec<usize>> = solutions.into_iter().map(|s| s.result()).collect();

    for (g_id, g) in graphs.into_iter().enumerate() {
        for (_nid, w) in g.node_weights().enumerate() {
            let original_nid = w.numerical_id;
            let pid = pid_arrays[g_id][original_nid];

            merged_pid_array[original_nid] = Some(pid + g_id * 2);
            //println!("(g_id, _nid, original_nid, pid) = ({}, {}, {}, {})", g_id, nid, original_nid, pid);
        }
    }
    
    //println!("{:?}", pid_arrays);
    //println!("{:?}", merged_pid_array);

    merged_pid_array
}

fn merge_pid_arrays(pid_arrays: &Vec<Vec<Option<usize>>>, graphs: &Vec<Graph<NodeInfo, usize, Directed, usize>>, num_nodes: usize, target_num_partitions: usize) -> Vec<Option<usize>>{
    let mut merged_pid_array: Vec<Option<usize>> = vec![None; num_nodes];

    for (g_id, g) in graphs.into_iter().enumerate() {
        for (_nid, w) in g.node_weights().enumerate() {
            let original_nid = w.numerical_id;
            let pid = pid_arrays[g_id][original_nid].unwrap();

            merged_pid_array[original_nid] = Some(pid + g_id * target_num_partitions / 2);
            //println!("(g_id, _nid, original_nid, pid) = ({}, {}, {}, {})", g_id, nid, original_nid, pid);
        }
    }
    
    //println!("{:?}", pid_arrays);
    //println!("{:?}", merged_pid_array);

    merged_pid_array
}

fn two_level_split_solve_merge(population_size: usize, num_generations: usize, target_num_partitions: usize, g: &Graph<NodeInfo, usize, Directed, usize>) {
    let core_bound = &vec![[0., 2.]; MAX_NUMBER_OF_NODES];
    let conf = RunConfig{probe_step_size: 1000, probe_step_vec: None};
    let mut partial_solutions: Vec<(usize, Vec<usize>)> = vec![];
    let _s = de_solve(&g, num_generations, population_size, None, None, true, core_bound, &conf, &mut partial_solutions);

    let pid_array = _s.result();
    let graph_vec = graph_splitter(&g, &pid_array);

    let mut solutions: Vec<Solver<BaseSolver<'_>>> = vec![];

    for i in 0..graph_vec.len() {
        let g_ref = &graph_vec[i];
        // TODO-PERFORMANCE: The sub-graphs might be processed much more quickly
        //                   by building shorter `core_bound`s for them.
        let sub_s = de_solve(g_ref, num_generations, population_size, None, None, true, core_bound, &conf, &mut partial_solutions);

        visualize_graph(g_ref, Some(&array_to_vec(&sub_s.result())), Some(format!("sub_graph_{}", i)));

        solutions.push(sub_s);
    }

    let merged_pid_array = merge_pid_arrays_from_solutions(&solutions, &graph_vec, g.node_count());
    println!("{:?}", merged_pid_array);
    visualize_graph(&g, Some(&merged_pid_array), Some(format!("merged_graph")));
}

fn expand_pid_array(g: &Graph<NodeInfo, usize, Directed, usize>, pid_array: &[usize], expanded_size: usize) -> Vec<usize> {
    let mut expanded_pid_array: Vec<usize> = vec![0; expanded_size];

    for (nid, w) in g.node_weights().enumerate() {
        let pid = pid_array[nid];
        let original_nid = w.numerical_id;

        expanded_pid_array[original_nid] = pid;
    }

    println!("{:?}", pid_array);
    println!("{:?}", expanded_pid_array);

    expanded_pid_array
}

// Here, pid_array is an optional pre-computed single-step optimization result
fn split_solve_merge(population_size: usize, num_generations: usize, target_num_partitions: usize, g: &Graph<NodeInfo, usize, Directed, usize>, pid_array: Option<&[usize]>) -> (Vec<Option<usize>>, Vec<(usize, Vec<usize>)>) {
    let mut core_bound = vec![[0., 2.]; MAX_NUMBER_OF_NODES];
    
    core_bound[0] = [0.,0.];
    
    //let core_bound = &vec![[0., 2.]; g.node_count()];
    let conf = RunConfig{probe_step_size: 200, probe_step_vec: Some(linspace_vec(1, num_generations, 8))};
    let mut partial_solutions: Vec<(usize, Vec<usize>)> = vec![];

    //let pid_array = expand_pid_array(&g, &_s.result(), MAX_NUMBER_OF_NODES);

    if target_num_partitions > 2 {
        //let graph_vec = graph_splitter(&g, &pid_array);
        let graph_vec = if pid_array.is_some() {
            graph_splitter(&g, &pid_array.unwrap())
        } else {
            let _s = de_solve(&g, num_generations, population_size, None, None, false, &core_bound, &conf, &mut partial_solutions);
            let pid_array = _s.result();
            graph_splitter(&g, &pid_array)
        };
        let mut partial_pid_arrays: Vec<Vec<Option<usize>>> = vec![];

        for i in 0..graph_vec.len() {
            let g_ref = &graph_vec[i];
            let (partial_pid_array, _) = split_solve_merge(population_size, (num_generations as f64 / 1.000) as usize, target_num_partitions / 2, g_ref, None);

            /*
            if target_num_partitions == MAX_NUMBER_OF_PARTITIONS {
                visualize_graph(g_ref, Some(&partial_pid_array), Some(format!("sub_graph_{}", i)));
            }
            */
            partial_pid_arrays.push(partial_pid_array);
        }

        let merged_pid_array = merge_pid_arrays(&partial_pid_arrays, &graph_vec, MAX_NUMBER_OF_NODES, target_num_partitions);
        //println!("{:?}", merged_pid_array);

        return (merged_pid_array, partial_solutions);
    } else {
        if pid_array.is_some() {
            return (array_to_vec(&pid_array.unwrap()), partial_solutions);
        } else {
            let conf = RunConfig{probe_step_size: usize::MAX, probe_step_vec: None};
            let _s = de_solve(&g, num_generations, population_size, None, None, false, &core_bound, &conf, &mut partial_solutions);
            let pid_array = _s.result();
            return (array_to_vec(&pid_array), partial_solutions);
        }
    }
}

fn local_search_surprise(graph: &Graph<NodeInfo, usize, Directed, usize>) -> Vec<usize> {
    let mut pid_array = vec![0; MAX_NUMBER_OF_NODES];
    let mut best_surprise_so_far = f64::MAX;
    let mut exhaustive_tries_since_improvement = 0;

    /*
    let mut rng = rand::thread_rng();
    for i in 0..MAX_NUMBER_OF_NODES {
        pid_array[i] = rng.gen_range(0..MAX_NUMBER_OF_PARTITIONS);
    }
    */

    'outer: loop {
        for i in 0..MAX_NUMBER_OF_NODES {
            let mut improved = false;
            let original_pid = pid_array[i];
            let mut best_pid = original_pid;

            for offset in 0..MAX_NUMBER_OF_PARTITIONS {
                let pid = (original_pid + offset) % MAX_NUMBER_OF_PARTITIONS;
                pid_array[i] = pid;

                let current_surprise = calculate_surprise(&graph, Some(&pid_array));

                if current_surprise < best_surprise_so_far {
                    improved = true;
                    best_surprise_so_far = current_surprise;
                    best_pid = pid;
                    println!("{}", current_surprise);
                }
            }

            if !improved {
                exhaustive_tries_since_improvement += 1;
            } else {
                exhaustive_tries_since_improvement = 0;
                pid_array[i] = best_pid;
            }

            if exhaustive_tries_since_improvement >= MAX_NUMBER_OF_NODES {
                break 'outer;
            }
        }
    }

    pid_array
}

fn local_search_permanence(graph: &Graph<NodeInfo, usize, Directed, usize>) -> Vec<usize> {
    let mut pid_array = vec![0; MAX_NUMBER_OF_NODES];
    let mut best_permanence_so_far = f64::MIN;
    let mut exhaustive_tries_since_improvement = 0;

    let mut rng = rand::thread_rng();
    for i in 0..MAX_NUMBER_OF_NODES {
        pid_array[i] = rng.gen_range(0..MAX_NUMBER_OF_PARTITIONS);
    }

    'outer: loop {
        println!("Starting main loop");
        for i in 0..MAX_NUMBER_OF_NODES {
            println!("[exhaustive_tries_since_improvement]: {}", exhaustive_tries_since_improvement);
            let mut improved = false;
            let original_pid = pid_array[i];
            let mut best_pid = original_pid;

            for offset in 0..MAX_NUMBER_OF_PARTITIONS {
                let pid = (original_pid + offset) % MAX_NUMBER_OF_PARTITIONS;
                pid_array[i] = pid;

                let current_permanence = calculate_permanence(&graph, &array_to_vec(&pid_array), &unwrap_pid_array(&array_to_vec(&pid_array)));

                if current_permanence > best_permanence_so_far {
                    improved = true;
                    best_permanence_so_far = current_permanence;
                    best_pid = pid;
                    println!("{}", current_permanence);
                }
            }

            if !improved {
                exhaustive_tries_since_improvement += 1;
            } else {
                exhaustive_tries_since_improvement = 0;
                pid_array[i] = best_pid;
            }

            if exhaustive_tries_since_improvement >= MAX_NUMBER_OF_NODES {
                break 'outer;
            }
        }
    }

    pid_array
}

fn unwrap_pid_array(pid_array: &[Option<usize>]) -> Vec<usize> {
    let mut unwrapped_pid_array: Vec<usize> = vec![];
    pid_array.into_iter().for_each(|x| unwrapped_pid_array.push(x.unwrap()));

    unwrapped_pid_array
}

fn test_multi_level_clustering(use_flattened_graph: bool) {
    let num_nodes = 64;
    let num_edges = 192;
    //let mixing_coeff = 0.003;
    let mixing_coeff = 0.000;
    let num_gen_communities = 8;
    let num_communities = 8;
    let max_comm_size_difference = 0;
    let num_flattening_passes = 2;

    let original_g = gen_lfr_like_graph(num_nodes, num_edges, mixing_coeff, num_gen_communities, max_comm_size_difference);
    let g = if use_flattened_graph {
        multi_pass_tree_transform(&original_g, 2, false)
    } else {
        original_g.clone()
    };
    println!("Finished generating random graph.");

    const TEMP_FILE_NAME: &str = "metaheuristics_evolution.csv";
    const NUM_SOLVER_EVALUATIONS: usize = 10;
    const NUM_SIMULATOR_EVALUATIONS: usize = 10;

    let mut best_surprise_per_algo = HashMap::new();

    let mut _f = File::create(TEMP_FILE_NAME).unwrap();
    const POP_SIZE: usize = 32;

    let num_gen_options = [12000];
    for num_generations in num_gen_options {
        for solver_iter in 0..NUM_SOLVER_EVALUATIONS {
            println!("\n\nSolver iteration: {}", solver_iter);
            //two_level_split_solve_merge(POP_SIZE, num_generations, num_communities, &g);
            let start = Instant::now();
            let (pid_array, partial_solutions) = split_solve_merge(POP_SIZE, num_generations, num_communities, &g, None);
            println!("Full multi-level solver elapsed time: {:?}", start.elapsed());

            let (immediate_successor_finalized_placements, immediate_successor_execution_info) = evaluate_execution_time_and_speedup(&original_g, &unwrap_pid_array(&pid_array), num_generations, true);
            let ims_permanence = calculate_permanence(&original_g, &immediate_successor_finalized_placements, &unwrap_pid_array(&immediate_successor_finalized_placements));
            let ims_surprise = calculate_surprise(&original_g, Some(&unwrap_pid_array(&immediate_successor_finalized_placements)));
            println!("Immediate succesor info: {:?}", immediate_successor_execution_info);
            println!("Immediate succesor permanence: {:?}", ims_permanence);
            println!("Immediate succesor surprise: {:?}", ims_surprise);

            for _ in 0..NUM_SIMULATOR_EVALUATIONS {
                let num_gen = 0;

                let (immediate_successor_finalized_placements, immediate_successor_execution_info) = evaluate_execution_time_and_speedup(&original_g, &unwrap_pid_array(&pid_array), num_gen, true);
                let ims_permanence = calculate_permanence(&original_g, &immediate_successor_finalized_placements, &unwrap_pid_array(&immediate_successor_finalized_placements));
                let ims_surprise = calculate_surprise(&original_g, Some(&unwrap_pid_array(&immediate_successor_finalized_placements)));

                _f.write(format!("Immediate successor,{},{},fitness_test,{},{},{},{},{},{}\n", num_gen, ims_surprise, immediate_successor_execution_info.speedup, KERNEL_NUMBER_OF_PARTITIONS, num_nodes, num_edges, num_communities, ims_permanence).as_bytes()).unwrap();
            }

            let algo_best = *best_surprise_per_algo.entry("Random Immediate Successor").or_insert(f64::MIN);
            if immediate_successor_execution_info.speedup > algo_best {
                visualize_graph(&original_g, Some(&immediate_successor_finalized_placements), Some(format!("immediate_successor_{}", immediate_successor_execution_info.speedup)));
                best_surprise_per_algo.insert("Random Immediate Successor", immediate_successor_execution_info.speedup);
            }

            let (finalized_core_placements, execution_info) = evaluate_execution_time_and_speedup(&original_g, &unwrap_pid_array(&pid_array), num_generations, false);
            let permanence = calculate_permanence(&original_g, &finalized_core_placements, &unwrap_pid_array(&finalized_core_placements));
            let surprise = calculate_surprise(&original_g, Some(&unwrap_pid_array(&finalized_core_placements)));
            println!("Multi-level solver info: {:?}", execution_info);
            println!("Multi-level solver permanence: {:?}", permanence);
            println!("Multi-level solver surprise: {:?}", surprise);

            let algo_best = *best_surprise_per_algo.entry("Multi-level differential evolution").or_insert(f64::MIN);
            if execution_info.speedup > algo_best {
                visualize_graph(&original_g, Some(&finalized_core_placements), Some(format!("merged_graph_{}", execution_info.speedup)));
                best_surprise_per_algo.insert("Multi-level differential evolution", execution_info.speedup);
            }

            let start = Instant::now();
            let tree_g = multi_pass_tree_transform(&original_g, num_flattening_passes, false);
            println!("Time for applying tree transformation {} times: {:?}", num_flattening_passes, start.elapsed());

            let start = Instant::now();
            let (tree_pid_array, tree_partial_solutions) = split_solve_merge(POP_SIZE, num_generations, num_communities, &tree_g, None);
            println!("Full multi-level tree solver elapsed time: {:?}", start.elapsed());

            let (finalized_core_placements, execution_info) = evaluate_execution_time_and_speedup(&original_g, &unwrap_pid_array(&tree_pid_array), num_generations, false);
            let permanence = calculate_permanence(&original_g, &finalized_core_placements, &unwrap_pid_array(&finalized_core_placements));
            let surprise = calculate_surprise(&original_g, Some(&unwrap_pid_array(&finalized_core_placements)));
            println!("Multi-level tree solver info: {:?}", execution_info);
            println!("Multi-level tree solver permanence: {:?}", permanence);
            println!("Multi-level tree solver surprise: {:?}", surprise);

            let algo_best = *best_surprise_per_algo.entry("Multi-level tree differential evolution").or_insert(f64::MIN);
            if execution_info.speedup > algo_best {
                visualize_graph(&original_g, Some(&finalized_core_placements), Some(format!("merged_graph_tree_{}", execution_info.speedup)));
                best_surprise_per_algo.insert("Multi-level tree differential evolution", execution_info.speedup);
            }

            for (num_gen, pid_array) in partial_solutions {
                let (pid_array, _) = split_solve_merge(POP_SIZE, num_gen, num_communities, &g, Some(&pid_array));

                for _ in 0..NUM_SIMULATOR_EVALUATIONS {
                    let (finalized_core_placements, execution_info) = evaluate_execution_time_and_speedup(&original_g, &unwrap_pid_array(&pid_array), num_gen, false);
                    let permanence = calculate_permanence(&original_g, &finalized_core_placements, &unwrap_pid_array(&finalized_core_placements));
                    let surprise = calculate_surprise(&original_g, Some(&unwrap_pid_array(&finalized_core_placements)));

                    _f.write(format!("Differential Evolution,{},{},fitness_test,{},{},{},{},{},{}\n", num_gen, surprise, execution_info.speedup, KERNEL_NUMBER_OF_PARTITIONS, num_nodes, num_edges, num_communities, permanence).as_bytes()).unwrap();
                }
            }

            for (num_gen, pid_array) in tree_partial_solutions {
                let (pid_array, _) = split_solve_merge(POP_SIZE, num_gen, num_communities, &tree_g, Some(&pid_array));

                for _ in 0..NUM_SIMULATOR_EVALUATIONS {
                    let (finalized_core_placements, execution_info) = evaluate_execution_time_and_speedup(&original_g, &unwrap_pid_array(&pid_array), num_gen, false);
                    let permanence = calculate_permanence(&original_g, &finalized_core_placements, &unwrap_pid_array(&finalized_core_placements));
                    let surprise = calculate_surprise(&original_g, Some(&unwrap_pid_array(&finalized_core_placements)));

                    _f.write(format!("Differential Evolution Tree,{},{},fitness_test,{},{},{},{},{},{}\n", num_gen, surprise, execution_info.speedup, KERNEL_NUMBER_OF_PARTITIONS, num_nodes, num_edges, num_communities, permanence).as_bytes()).unwrap();
                }
            }
        }
    }

    visualize_graph(&multi_pass_tree_transform(&original_g, num_flattening_passes, false), None, Some("ground_truth_flattened".to_string()));
    visualize_graph(&original_g, None, Some("ground_truth_original".to_string()));
    gen_speedup_bars(TEMP_FILE_NAME, "speedups");
}

fn linspace_vec(min_val: usize, max_val: usize, num_elements: usize) -> Vec<usize> {
    let mut v: Vec<usize> = Vec::with_capacity(num_elements);

    v.push(min_val);

    let min_val = if min_val < 1 {
        1
    } else {
        min_val
    };

    let min_l = (min_val as f64).log2();
    let max_l = (max_val as f64).log2();
    let step_size = (max_l - min_l) / ((num_elements - 1) as f64);

    for i in 1..(num_elements - 1) {
        v.push(f64::exp2(min_l + step_size * (i as f64)).round() as usize);
    }

    v.push(max_val);

    v
}

fn tree_transform(original_graph: &Graph<NodeInfo, usize, Directed, usize>, adapt_weights: bool) -> Graph<NodeInfo, usize, Directed, usize> {
    let mut g = original_graph.clone();
    let mut was_added = vec![false; MAX_NUMBER_OF_NODES];

    g.clear_edges();

    let node_refs: Vec<(petgraph::prelude::NodeIndex<usize>, &NodeInfo)> = original_graph.node_references().collect();
    let first_node = node_refs[0].id();
    let mut parent_vec: Vec<Option<petgraph::prelude::NodeIndex<usize>>> = vec![None; MAX_NUMBER_OF_NODES];

    let mut node_queue = vec![first_node];    

    'outer: loop {
        while !node_queue.is_empty() {
            let v = node_queue.pop().unwrap();
            let v_id = original_graph.node_weight(v).unwrap().numerical_id;
            was_added[v_id] = true;

            for n in original_graph.neighbors_undirected(v) {
                let n_id = original_graph.node_weight(n).unwrap().numerical_id;

                if !was_added[n_id] {
                    was_added[n_id] = true;
                    parent_vec[n_id] = Some(v);
                    node_queue.push(n);
                }
            }
        }

        for v in original_graph.node_references() {
            let v_id = v.1.numerical_id;

            if !was_added[v_id] {
                node_queue.push(v.id());
                continue 'outer;
            }
        }

        break;
    }

    for i in 0..node_refs.len() {
        let pseudo_child = node_refs[i].id();

        if parent_vec[i].is_some() {
            let pseudo_parent = parent_vec[i].unwrap();

            if original_graph.contains_edge(pseudo_parent, pseudo_child) {
                g.add_edge(pseudo_parent, pseudo_child, 1);
            } else {
                g.add_edge(pseudo_child, pseudo_parent, 1);
            }
        }
    }

    if adapt_weights {
        let mut moved_weights = vec![0; MAX_NUMBER_OF_NODES];
        for e in original_graph.edge_references() {
            /*
            println!("{:?}", e);
            println!("{:?}", e.source());
            println!("{:?}", e.target());
            println!("Found in flattened graph? {:?}", g.contains_edge(e.source(), e.target()));
            println!();
            */

            let s = e.source();
            let t = e.target();

            if !g.contains_edge(s, t) {
                let s_id = original_graph.node_weight(s).unwrap().numerical_id;
                let edge_ref = original_graph.find_edge(s, t).unwrap();
                let e_weight = original_graph.edge_weight(edge_ref).unwrap();

                moved_weights[s_id] += 1;
                //moved_weights[s_id] += *e_weight;
            }
        }

        // TODO: PERFORMANCE
        //       Avoid the extra pass for handling weighted
        //       nodes with no outgoing edges.

        for v in g.node_references() {
            let v_id = g.node_weight(v.0).unwrap().numerical_id;
            if moved_weights[v_id] > 0 && g.neighbors_directed(v.0, petgraph::Outgoing).count() == 0 {
                // TODO: PERFORMANCE
                // TODO: Prove that we are always able to escape from this loop
                'outer: loop {
                    for a in g.neighbors_directed(v.0, petgraph::Incoming) {
                        if moved_weights[v_id] > 0 {
                            let a_id = g.node_weight(a.id()).unwrap().numerical_id;
                            moved_weights[a_id] += 1;
                            moved_weights[v_id] -= 1;
                        } else {
                            break 'outer;
                        }
                    }

                }
            }
        }

        //println!("{:?}", moved_weights);
        let mut weighted_g = g.clone();

        for v in g.node_references() {
            let v_id = g.node_weight(v.0).unwrap().numerical_id;
            if moved_weights[v_id] > 0 {
                'outer: loop {
                    for a in g.neighbors_directed(v.0, petgraph::Outgoing) {
                        if moved_weights[v_id] > 0 {
                            let edge = weighted_g.edges_connecting(v.0, a.id()).last().unwrap();
                            let edge_weight = weighted_g.edge_weight_mut(edge.id()).unwrap();
                            *edge_weight = *edge_weight + 1;
                            moved_weights[v_id] -= 1;
                        } else {
                            break 'outer;
                        }
                    }

                }
            }
        }

        //println!("{:?}", moved_weights);
        return weighted_g;
    }

    g
}

fn multi_pass_tree_transform(original_graph: &Graph<NodeInfo, usize, Directed, usize>, num_passes: usize, adapt_weights: bool) -> Graph<NodeInfo, usize, Directed, usize> {
    let mut next_pass: Graph<NodeInfo, usize, Directed, usize> = tree_transform(&original_graph, adapt_weights);

    for _ in 1..num_passes {
        let last_pass = next_pass.clone();

        next_pass = original_graph.clone();
        for e in last_pass.edge_references() {
            let e_ref = next_pass.find_edge(e.source(), e.target()).unwrap();
            next_pass.remove_edge(e_ref);
        }

        next_pass = tree_transform(&next_pass, adapt_weights);
        for e in last_pass.edge_references() {
            next_pass.add_edge(e.source(), e.target(), *e.weight());
        }
    }

    next_pass
}

fn test_tree_transform() {
    let num_nodes = 64;
    let num_edges = 192;
    let mixing_coeff = 0.001;
    let num_communities = 8;
    let max_comm_size_difference = 0;

    let g = gen_lfr_like_graph(num_nodes, num_edges, mixing_coeff, num_communities, max_comm_size_difference);
    visualize_graph(&g, None, Some("original_graph".to_string()));
    //println!("{:?}", &g);

    let g_tree = tree_transform(&g, false);
    visualize_graph(&g_tree, None, Some("tree_transformed_graph".to_string()));
    //println!("{:?}", &g_tree);
}

fn test_n_tree_transform() {
    let num_nodes = 64;
    let num_edges = 192;
    let mixing_coeff = 0.001;
    let num_communities = 8;
    let max_comm_size_difference = 0;

    let g = gen_lfr_like_graph(num_nodes, num_edges, mixing_coeff, num_communities, max_comm_size_difference);
    visualize_graph(&g, None, Some("original_graph".to_string()));
    println!("{:?}", &g);

    let g_tree = multi_pass_tree_transform(&g, 1, false);
    visualize_graph(&g_tree, None, Some("tree_transformed_graph_1_pass".to_string()));
    println!("{:?}", &g_tree);

    let g_tree = multi_pass_tree_transform(&g, 2, false);
    visualize_graph(&g_tree, None, Some("tree_transformed_graph_2_passes".to_string()));
    println!("{:?}", &g_tree);

    let g_tree = multi_pass_tree_transform(&g, 3, false);
    visualize_graph(&g_tree, None, Some("tree_transformed_graph_3_passes".to_string()));
    println!("{:?}", &g_tree);
}

fn test_local_surprise_search() {
    let num_nodes = 64;
    let num_edges = 192;
    let mixing_coeff = 0.000;
    let num_communities = 8;
    let max_comm_size_difference = 0;

    let g = gen_lfr_like_graph(num_nodes, num_edges, mixing_coeff, num_communities, max_comm_size_difference);
    //println!("{:?}", &g);

    //let g_tree = multi_pass_tree_transform(&g, 1, false);
    //println!("{:?}", &g_tree);

    let start = Instant::now();
    let pid_array_0 = local_search_surprise(&g);
    println!("Time for completing local search: {:?}", start.elapsed());
    visualize_graph(&g, Some(&array_to_vec(&pid_array_0)), Some(format!("local_search_surprise_0")));
    //visualize_graph(g_ref, Some(&array_to_vec(&sub_s.result())), Some(format!("sub_graph_{}", i)));
}

fn test_local_permanence_search() {
    let num_nodes = 64;
    let num_edges = 64;
    let mixing_coeff = 0.000;
    let num_communities = 8;
    let max_comm_size_difference = 0;

    let g = gen_lfr_like_graph(num_nodes, num_edges, mixing_coeff, num_communities, max_comm_size_difference);
    //println!("{:?}", &g);

    //let g_tree = multi_pass_tree_transform(&g, 1, false);
    //println!("{:?}", &g_tree);

    let start = Instant::now();
    let pid_array_0 = local_search_permanence(&g);
    println!("Time for completing local search: {:?}", start.elapsed());
    visualize_graph(&g, Some(&array_to_vec(&pid_array_0)), Some(format!("local_search_permanence_0")));
    //visualize_graph(g_ref, Some(&array_to_vec(&sub_s.result())), Some(format!("sub_graph_{}", i)));
}

fn test_dinitz_max_flow() {
    let mut rng = rand::thread_rng();

    const NUM_NODES: usize = 256;
    //const NUM_EDGES: usize = 8 * NUM_NODES;
    const NUM_EDGES: usize = 2 * NUM_NODES - 2;
    const SOURCE: usize = 0;
    const SINK: usize = NUM_NODES;
    //const MAX_WEIGHT: usize = NUM_NODES;
    const MAX_WEIGHT: usize = 5;
    const NUM_SOLVER_EVALUATIONS: usize = NUM_NODES;
    const ALPHA: usize = 1;

    let mut s_vec: Vec<usize> = vec![];
    let mut t_vec: Vec<usize> = vec![];
    let mut w_vec: Vec<i32> = vec![];

    for _ in 0..NUM_EDGES {
        let s = rng.gen_range(SOURCE..=SINK - 1);
        let t = rng.gen_range(SOURCE..=SINK - 1);
        let w = rng.gen_range(1..=MAX_WEIGHT);

        s_vec.push(s);
        t_vec.push(t);
        w_vec.push(w as i32);
    }

    let mut total_compute_time = Duration::ZERO;

    for _ in 0..NUM_SOLVER_EVALUATIONS {
        let start = Instant::now();
        let mut flow: DinicMaxFlow<i32> = DinicMaxFlow::new(SOURCE, SINK, NUM_NODES + 1);

        for i in 0..NUM_EDGES {
            flow.add_edge(s_vec[i], t_vec[i], w_vec[i]);
        }

        for i in 0..NUM_NODES {
            flow.add_edge(i, SINK, ALPHA as i32);
        }
        println!("Time for building graph: {:?}", start.elapsed());

        let start = Instant::now();
        let max_flow = flow.find_maxflow(i32::MAX);

        let compute_time = start.elapsed();
        total_compute_time += compute_time;
        println!("Time for calculating max flow: {:?}", compute_time);

        println!("Max flow: {}", max_flow);
    }
    println!("Total time for computing max-flow {NUM_SOLVER_EVALUATIONS} times: {:?}", total_compute_time);
}

fn build_dinitz_instance<T: EdgeType>(graph: &Graph<NodeInfo, usize, T, usize>, source: usize, sink: usize) -> DinicMaxFlow<i64> {
    let num_nodes = graph.node_count();
    let mut flow: DinicMaxFlow<i64> = DinicMaxFlow::new(source, sink, num_nodes);

    for e in graph.edge_references() {
        let a = e.source();
        let b = e.target();
        let w = e.weight();
        let a_id = graph.node_weight(a).unwrap().numerical_id;
        let b_id = graph.node_weight(b).unwrap().numerical_id;
        //println!("[build_dinitz_instance]: (a, b, w): ({}, {}, {})", a, b, *w);

        flow.add_edge(a_id, b_id, *w as i64);
        flow.add_edge(b_id, a_id, *w as i64);
    }

    flow
}

fn get_node_sets_from_min_cut(cut_tree: &GraphMap<NodeInfo, usize, Undirected>, flow_edges: &Vec<FlowResultEdge<i64>>, source_node_id: usize, compute_sink_node_set: bool) -> Vec<HashSet<i64>> {
    let mut node_sets: Vec<HashSet<i64>> = Vec::with_capacity(2);

    let start = Instant::now();
    //TODO-PERFORMANCE: Avoid cloning the tree here, since this function is called every time
    //                  max_flow needs to be calculated by `gusfield_gomory_hu_solver`
    let mut cut_tree = cut_tree.clone();
    println!("Time for cloning cut tree prior to cutting: {:?}", start.elapsed());
    //println!("[get_node_sets_from_min_cut]: flow_edges: {:?}", flow_edges);

    for e in flow_edges {
        let s = e.source;
        let t = e.sink;
        let flow = e.flow;

        let s_info = NodeInfo{numerical_id: s, partition_id: 0};
        let t_info = NodeInfo{numerical_id: t, partition_id: 0};

        let capacity = cut_tree.edge_weight(s_info, t_info);

        if capacity.is_some() {
            let capacity = *capacity.unwrap() as i64;
            if flow == (capacity as i64) {
                cut_tree.remove_edge(NodeInfo{numerical_id: s, partition_id: 0}, NodeInfo{numerical_id: t, partition_id: 0});
            }
        }
    }

    //visualize_graph(&cut_tree.clone().into_graph(), None, Some("cut_tree_after_cutting".to_string()));

    //Running the following line costs some 500 ns if there is a single edge to print
    //println!("[get_node_sets_from_min_cut]: flow_edges: {:?}", flow_edges);

    //let node_refs: Vec<(NodeInfo, &NodeInfo)> = cut_tree.node_references().collect();
    // The first node should be the flow src, not any vertex
    //let first_node = node_refs[0].id();
    let first_node = NodeInfo{numerical_id: source_node_id, partition_id: 0};
    let mut parent_vec: Vec<Option<NodeInfo>> = vec![None; MAX_NUMBER_OF_NODES];

    let mut node_queue = vec![first_node];    

    // The version based on the BTreeMap has lower complexity,
    // but it performs worse than the simpler one for graphs
    // with 50K nodes or less.
    let mut was_added = vec![false; MAX_NUMBER_OF_NODES];
    let mut set_id = 0;
    node_sets.push(HashSet::new());
    
    while !node_queue.is_empty() {
        let v = node_queue.pop().unwrap();
        let v_id = v.numerical_id;
        was_added[v_id] = true;
        node_sets[set_id].insert(v_id as i64);

        for n in cut_tree.neighbors(v) {
            let n_id = n.numerical_id;

            if !was_added[n_id] {
                was_added[n_id] = true;
                parent_vec[n_id] = Some(v);
                node_queue.push(n);
            }
        }
    }

    // Calculating the node set not containing the flow source node
    // is not needed by Gusfield's algorithm, so we can frequently
    // avoid this to reduce compute time.
    if compute_sink_node_set {
        node_sets.push(HashSet::new());
        set_id += 1;

        for v in cut_tree.node_references() {
            let v_id = v.1.numerical_id;

            if !was_added[v_id] {
                node_sets[set_id].insert(v.id().numerical_id as i64);
            }
        }
        //println!("[get_node_sets_from_min_cut]: node_sets: {:?}", node_sets);
    }

    node_sets
}

fn get_max_flow_and_node_sets(cut_tree: &GraphMap<NodeInfo, usize, Undirected>, flow_instance: &mut DinicMaxFlow<i64>, source_node_id: usize, compute_sink_node_set: bool) -> (i64, Vec<HashSet<i64>>) {
    let max_flow = flow_instance.find_maxflow(i64::MAX);
    let flow_edges = flow_instance.get_flow_edges(i64::MAX);
    //let (node_set_x, node_set_y) = get_node_sets_from_min_cut(&cut_tree, &flow_edges);
    let node_sets = get_node_sets_from_min_cut(&cut_tree, &flow_edges, source_node_id, compute_sink_node_set);
    //assert!(node_sets.len() == 2);

    (max_flow, node_sets)
}

// Implemented according to Gusfield, Dan. "Very simple methods for all pairs network flow analysis."
// SIAM Journal on Computing 19.1 (1990): 143-155.
// https://doi.org/10.1137/0219009
//
// This implementation ignores directionality and will output an undirected tree.
#[allow(dead_code)]
fn gusfield_gomory_hu_solver<T: EdgeType>(graph: &Graph<NodeInfo, usize, T, usize>) -> GraphMap<NodeInfo, usize, Undirected> {
    // TODO-PERFORMANCE: We should find a way to quickly clone DinicMaxFlow objects
    //                   rather than spend time rebuilding them from external data.
                       
    //println!("[gusfield_gomory_hu_solver]: Visualizing reference graph");
    //visualize_graph(&graph, None, Some("reference_graph".to_string()));

    let mut cut_tree = GraphMap::<NodeInfo, usize, Undirected>::new();
    let num_nodes = graph.node_count();

    for i in 0..num_nodes {
        cut_tree.add_node(NodeInfo{numerical_id: i, partition_id: 0});
    }

    for t in 1..num_nodes {
        let s = 0;
        let w = 1;

        cut_tree.add_edge(NodeInfo{numerical_id: s, partition_id: 0}, NodeInfo{numerical_id: t, partition_id: 0}, w);
    }
    //visualize_graph(&cut_tree.clone().into_graph(), None, Some("cut_tree_initial_state".to_string()));

    let mut first = true;
    for s in 1..num_nodes {
        let s_info = NodeInfo{numerical_id: s, partition_id: 0};

        println!("\ns: {s}");
        let start = Instant::now();
        let t = cut_tree.neighbors(s_info).next().unwrap().numerical_id;
        let t_info = NodeInfo{numerical_id: t, partition_id: 0};
        println!("Time for finding t: {:?}", start.elapsed());

        let source = s;
        let sink = t;

        let start = Instant::now();
        //let mut flow_instance: DinicMaxFlow<i64> = build_dinitz_instance(&cut_tree, source, sink);
        let mut flow_instance: DinicMaxFlow<i64> = build_dinitz_instance(&graph, source, sink);
        println!("Time for building dinitz instance: {:?}", start.elapsed());
        let start = Instant::now();

        let mut graph_map = GraphMap::<NodeInfo, usize, Undirected>::new();
        for n in graph.node_indices() {
            graph_map.add_node(NodeInfo{numerical_id: n.index(), partition_id: 0});
        }

        for e in graph.edge_references() {
            let a = e.source();
            let b = e.target();
            let w = e.weight();
            let a_id = graph.node_weight(a).unwrap().numerical_id;
            let b_id = graph.node_weight(b).unwrap().numerical_id;

            graph_map.add_edge(NodeInfo{numerical_id: a_id, partition_id: 0}, NodeInfo{numerical_id: b_id, partition_id: 0}, *w);
        }

        let (max_flow, node_sets) = get_max_flow_and_node_sets(&graph_map, &mut flow_instance, source, false);
        println!("Time for calculating max flow: {:?}", start.elapsed());

        let edge_to_label = cut_tree.edge_weight_mut(s_info, t_info).unwrap();
        *edge_to_label = max_flow as usize;
        //println!("max_flow({s}, {t}): {max_flow}");

        //if first && max_flow != 0 {
        if first {
            first = false;
            //println!("{:?}", cut_tree);
            //visualize_graph(&cut_tree.clone().into_graph(), None, Some("cut_tree_flow_graph".to_string()));
        }
        
        //println!("node_sets: {:?}", node_sets);

        for i in 0..num_nodes {
            if i == s {
                continue;
            }
            
            let i_info = NodeInfo{numerical_id: i, partition_id: 0};
            //println!("(s, i, t) = ({s}, {i}, {t})");

            let set = &node_sets[0];
            if set.contains(&(s as i64)) && set.contains(&(i as i64)) && cut_tree.contains_edge(i_info, t_info) {
                //println!("Removing ({i}, {t}) and adding ({s}, {i})");
                let old_label = cut_tree.remove_edge(i_info, t_info).unwrap();
                cut_tree.add_edge(s_info, i_info, old_label);
            }
        }
    }

    //let cut_tree: Graph<NodeInfo, usize, Undirected, usize> = cut_tree.into_graph();
    cut_tree
}

#[allow(dead_code)]
fn text_export_graph<T: EdgeType>(graph: &Graph<NodeInfo, usize, T, usize>) {
    println!("{} {}", graph.node_count(), graph.edge_count());

    for e in graph.edge_references() {
        let s = e.source().index();
        let t = e.target().index();
        let w = e.weight().index();

        println!("{:?} {:?} {:?}", s, t, w);
    }
}

#[allow(dead_code)]
fn generate_max_flow_clustering_graph<T: EdgeType>(graph: &Graph<NodeInfo, usize, T, usize>, alpha: usize, weight_override: Option<usize>) -> Graph<NodeInfo, usize, T, usize> {
    let mut instance = graph.clone();
    let original_node_count = graph.node_count();

    let sink = instance.add_node(NodeInfo{numerical_id: original_node_count, partition_id: 0});
    
    let mut num_added_edges = 0;
    let node_refs: Vec<NodeIndex<usize>> = instance.node_references().map(|(a, _)| a).collect();

    if weight_override.is_some() {
        let w = weight_override.unwrap();

        for e in instance.edge_weights_mut() {
            *e = w;
        }
    }

    for n in node_refs {
        if num_added_edges >= original_node_count {
            break;
        }

        instance.add_edge(sink, n.id(), alpha);
        num_added_edges += 1;
    }

    instance
}

fn export_clustering_problem() {
    let num_nodes = 256;
    let num_edges = num_nodes * 2 - 2;
    //let mixing_coeff = 0.003;
    let mixing_coeff = 0.000;
    let num_gen_communities = 8;
    let max_comm_size_difference = 0;

    let graph: Graph<NodeInfo, usize, Undirected, usize> = gen_lfr_like_graph(num_nodes, num_edges, mixing_coeff, num_gen_communities, max_comm_size_difference);

    text_export_graph(&generate_max_flow_clustering_graph(&graph, 1, Some(5)));
    //text_export_graph(&graph, Some(5));
}

fn import_text_graph() -> Graph<NodeInfo, usize, Directed, usize> {
   let (num_nodes, num_edges) = scanln_fmt!( "{d} {d}", usize, usize).unwrap();
   //println!("Got {} and {}", num_nodes, num_edges);

    let mut g = GraphMap::<NodeInfo, usize, Directed>::with_capacity(num_nodes, num_edges);

    for i in 0..num_nodes {
        g.add_node(NodeInfo{numerical_id: i, partition_id: 0});
    }

    for _ in 0..num_edges {
       let (s, t, w) = scanln_fmt!( "{d} {d} {d}", usize, usize, usize).unwrap();
       //println!("Got {} and {}", num_nodes, num_edges);

       g.add_edge(NodeInfo{numerical_id: s, partition_id: 0}, NodeInfo{numerical_id: t, partition_id: 0}, w);
    }

    let g: Graph<NodeInfo, usize, Directed, usize> = g.into_graph();
    g
}

fn import_and_visualize_graph() {
    let imported_graph = import_text_graph();
    visualize_graph(&imported_graph, None, Some("imported_max_flow_clustering".to_string()));
}

fn get_node_vecs<T: EdgeType>(graph: &GraphMap<NodeInfo, usize, T>) -> Vec<Vec<usize>> {
    let mut node_vecs = vec![];

    let node_refs: Vec<(NodeInfo, &NodeInfo)> = graph.node_references().collect();
    let first_node = node_refs[0].id();
    let mut parent_vec: Vec<Option<NodeInfo>> = vec![None; MAX_NUMBER_OF_NODES];

    let mut node_queue = vec![first_node];    

    let mut was_added = vec![false; MAX_NUMBER_OF_NODES];
    let mut set_id = 0;
    'outer: loop {
        node_vecs.push(vec![]);
        
        while !node_queue.is_empty() {
            let v = node_queue.pop().unwrap();
            let v_id = v.numerical_id;
            was_added[v_id] = true;
            node_vecs[set_id].push(v_id);

            for n in graph.neighbors(v) {
                let n_id = n.numerical_id;

                if !was_added[n_id] {
                    was_added[n_id] = true;
                    parent_vec[n_id] = Some(v);
                    node_queue.push(n);
                }
            }
        }
        set_id += 1;

        for v in graph.node_references() {
            let v_id = v.1.numerical_id;

            if !was_added[v_id] {
                node_queue.push(v.id());
                continue 'outer;
            }
        }

        break;
    }

    node_vecs
}

//TODO
fn max_flow_cluster<T: EdgeType>(graph: &Graph<NodeInfo, usize, T, usize>, alpha: usize, weight_override: Option<usize>) -> Vec<usize> {
    let mut pid_array = vec![];
    pid_array.resize(graph.node_count(), 0);
    //let weight = weight_override.unwrap_or(1);

    //let cut_tree = gusfield_gomory_hu_solver(&graph);
    //visualize_graph(&cut_tree.into_graph(), None, Some(format!("max_flow_reference_cut_tree").to_string()));

    let clustering_instance = generate_max_flow_clustering_graph(graph, alpha, weight_override);
    //visualize_graph(&clustering_instance, None, Some(format!("max_flow_clustering_instance_alpha{alpha}").to_string()));
    let mut cut_tree = gusfield_gomory_hu_solver(&clustering_instance);

    let fake_sink_nid = cut_tree.node_count() - 1;
    let fake_sink = cut_tree.node_references().find(|x| x.1.numerical_id == fake_sink_nid).unwrap().id();

    //let num_nodes = cut_tree.node_count() - 1;
    //visualize_graph(&cut_tree.clone().into_graph(), None, Some(format!("max_flow_clustering_cut_tree_with_fake_sink_n{num_nodes}_alpha{alpha}_weight{weight}").to_string()));

    cut_tree.remove_node(fake_sink);

    //let num_nodes = cut_tree.node_count();
    //visualize_graph(&cut_tree.clone().into_graph(), None, Some(format!("max_flow_clustering_cut_tree_n{num_nodes}_alpha{alpha}_weight{weight}").to_string()));

    let node_vecs = get_node_vecs(&cut_tree);

    for (i, node_vec) in node_vecs.into_iter().enumerate() {
        for nid in node_vec {
            pid_array[nid] = i;
        }
    }

    pid_array
}

pub fn test_gusfield_gomory_hu_solver() {
    let num_nodes = 256;
    let num_edges = num_nodes * 2 - 2;
    //let mixing_coeff = 0.003;
    let mixing_coeff = 0.000;
    let num_gen_communities = 8;
    let max_comm_size_difference = 0;

    let graph: Graph<NodeInfo, usize, Undirected, usize> = gen_lfr_like_graph(num_nodes, num_edges, mixing_coeff, num_gen_communities, max_comm_size_difference);
    println!("Finished generating random graph.");
    let start = Instant::now();
    let cut_tree = gusfield_gomory_hu_solver(&graph);
    println!("Total Gusfield Gomory Hu solver time: {:?}", start.elapsed());

    visualize_graph(&cut_tree.into_graph(), None, Some("gusfield_gomory_hu_cut_tree".to_string()));
}

pub fn test_max_flow_clustering() {
    let num_nodes = 64;
    let num_edges = num_nodes * 2 - 2;
    //let mixing_coeff = 0.003;
    let mixing_coeff = 0.002;
    let num_gen_communities = 8;
    let max_comm_size_difference = 0;
    let alpha = 1;
    let weight = 15;

    let graph: Graph<NodeInfo, usize, Undirected, usize> = gen_lfr_like_graph(num_nodes, num_edges, mixing_coeff, num_gen_communities, max_comm_size_difference);
    visualize_graph(&graph, None, Some(format!("max_flow_clustering_reference_n{num_nodes}_alpha{alpha}_weight{weight}_mixing{mixing_coeff}_num_comm{num_gen_communities}").to_string()));

    let start = Instant::now();
    let pid_array = max_flow_cluster(&graph, alpha, Some(weight));
    println!("[test_max_flow_clustering]: Time for running max_flow_cluster: {:?}", start.elapsed());
    println!("[test_max_flow_clustering]: pid_array: {:?}", pid_array);

    visualize_graph(&graph, Some(&array_to_vec(&pid_array)), Some(format!("max_flow_clustered_graph_n{num_nodes}_alpha{alpha}_weight{weight}_mixing{mixing_coeff}_num_comm{num_gen_communities}").to_string()));
    println!("[test_max_flow_clustering]: Test finished.");

    //text_export_graph(&generate_max_flow_clustering_graph(&graph, 1, Some(5)));
}

// TODO-PERFORMANCE
// Let DepGraph be setup in a way that is optimized
// for contiguous task IDs, which reduce the need
// for hash-based data structures.
// Take benefit of the commented code for that.
pub struct DepGraph {
    writer_task_per_dep: FxHashMap<usize, usize>,
    reader_tasks_per_dep: FxHashMap<usize, FxHashSet<usize>>,

    // TODO-PERFORMANCE
    // If we really wanted to extract as much performance
    // as possible from this, we should implement our own
    // HashMaps and arrays to avoid memory reallocations
    // as much as possible and use faster hash functions.
    //write_deps_per_task: Vec<Vec<usize>>,
    //read_deps_per_task: Vec<Vec<usize>>,
    write_deps_per_task: FxHashMap<usize, Vec<usize>>,
    read_deps_per_task: FxHashMap<usize, Vec<usize>>,
    task_dependency_graph: GraphMap::<NodeInfo, usize, Directed>,
    ready_tasks: Vec<usize>,
    dep_counter_per_task: FxHashMap<usize, usize>,
    use_adj_matrix: bool,
}

// TODO: Add checks for ensuring that we
//       block submission when we have
//       MAX_NUM_TASKS in-flight tasks.
//const MAX_NUM_TASKS: usize = 100000;

impl DepGraph {
    pub fn new(use_matrix: bool) -> Self {

        let obj = Self {
            writer_task_per_dep: FxHashMap::default(),
            reader_tasks_per_dep: FxHashMap::default(),
            //write_deps_per_task: Vec::with_capacity(MAX_NUM_TASKS),
            //read_deps_per_task: Vec::with_capacity(MAX_NUM_TASKS),
            write_deps_per_task: FxHashMap::default(),
            read_deps_per_task: FxHashMap::default(),
            task_dependency_graph: GraphMap::<NodeInfo, usize, Directed>::new(),
            ready_tasks: vec![],
            //dep_counter_per_task: vec![0; MAX_NUM_TASKS],
            dep_counter_per_task: FxHashMap::default(),
            use_adj_matrix: use_matrix,
        };

        /*
        for _ in 0..MAX_NUM_TASKS {
            obj.write_deps_per_task.push(vec![]);
            obj.read_deps_per_task.push(vec![]);
        }
        */

        obj
    }

    pub fn add_task_write_dep(&mut self, task: usize, dep: usize) {
        /*
        let deps_vec = self.write_deps_per_task.get_mut(&task).unwrap();
        deps_vec.push(dep);
        */

        if let Some(deps_vec) = self.write_deps_per_task.get_mut(&task) {
            deps_vec.push(dep);
        } else {
            self.read_deps_per_task.insert(task, vec![dep]);
        }

        // We use "mut" here to indicate that
        // the closure is not idempotent
        // https://doc.rust-lang.org/std/ops/trait.FnMut.html
        let mut make_descendant_of_task = |dep_holder_task: usize| {
            if let Some(counter) = self.dep_counter_per_task.get_mut(&task) {
                *counter += 1;
            } else {
                self.dep_counter_per_task.insert(task, 1);
            }

            if self.use_adj_matrix {
                let dep_holder_task = NodeInfo::new(dep_holder_task);
                let task = NodeInfo::new(task);

                let old_w = self.task_dependency_graph.edge_weight(dep_holder_task, task);

                if let Some(old_w) = old_w {
                    self.task_dependency_graph.add_edge(dep_holder_task, task, *old_w + 1);
                } else {
                    self.task_dependency_graph.add_edge(dep_holder_task, task, 1);
                }
            }
        };

        if let Some(reader_tasks_set) = self.reader_tasks_per_dep.get_mut(&dep) {
            for reader_task in reader_tasks_set.iter() {
                make_descendant_of_task(*reader_task);
            }

            reader_tasks_set.clear();

            assert!(self.reader_tasks_per_dep.get_mut(&dep).unwrap().is_empty(), "reader_task_vec was not cleared");
        }

        let writer_task = self.writer_task_per_dep.get_mut(&dep);

        if writer_task.is_some() {
            let writer_task = writer_task.unwrap();
            make_descendant_of_task(*writer_task);
            *writer_task = task;
        } else {
            self.writer_task_per_dep.insert(dep, task);
        }
    }

    // TODO: Develop some way of ensuring that the same dependence
    //       is not added as both a read and a write dependence
    pub fn add_task_read_dep(&mut self, task: usize, dep: usize) {
        /*
        let deps_vec = self.read_deps_per_task.get_mut(&task).unwrap();
        deps_vec.push(dep);
        */

        if let Some(deps_vec) = self.read_deps_per_task.get_mut(&task) {
            deps_vec.push(dep);
        } else {
            self.read_deps_per_task.insert(task, vec![dep]);
        }

        if let Some(owner_task) = self.writer_task_per_dep.get(&dep) {
            if let Some(counter) = self.dep_counter_per_task.get_mut(&task) {
                *counter += 1;
            } else {
                self.dep_counter_per_task.insert(task, 1);
            }

            if self.use_adj_matrix {
                let owner_task = NodeInfo::new(*owner_task);
                let task = NodeInfo::new(task);

                let old_w = self.task_dependency_graph.edge_weight(owner_task, task);

                if let Some(old_w) = old_w {
                    self.task_dependency_graph.add_edge(owner_task, task, *old_w + 1);
                } else {
                    self.task_dependency_graph.add_edge(owner_task, task, 1);
                }
            }
        }

        let reader_tasks_set = self.reader_tasks_per_dep.get_mut(&dep);

        if reader_tasks_set.is_some() {
            let reader_tasks_set = reader_tasks_set.unwrap();
            reader_tasks_set.insert(task);
        } else {
            let mut reader_tasks_set = FxHashSet::default();
            reader_tasks_set.reserve(7);
            self.reader_tasks_per_dep.insert(dep, reader_tasks_set);
        }
    }

    pub fn add_task(&mut self, task: usize) {
        if self.use_adj_matrix {
            let task = NodeInfo::new(task);
            self.task_dependency_graph.add_node(task);
        }
    }

    pub fn finish_adding_task(&mut self, task: usize) {
        // TODO-PERFORMANCE
        // We could avoid checking this for tasks without
        // dependencies by obtaining more information of
        // task type from the caller.
        let counter = self.dep_counter_per_task.get(&task).unwrap_or(&0);
        if *counter == 0 {
            self.ready_tasks.push(task);
        }
    }

    pub fn retire_task(&mut self, task: usize) {
        // # Read deps
        // Trigger changes to depending tasks
        if let Some(read_deps) = self.read_deps_per_task.get(&task) {
            for r_dep in read_deps {
                if let Some(war_dependent_task) = self.writer_task_per_dep.get(&r_dep) {
                    let counter = self.dep_counter_per_task.get_mut(war_dependent_task).unwrap();
                    *counter -= 1;
                }

                if let Some(read_set) = self.reader_tasks_per_dep.get_mut(&r_dep) {
                    read_set.remove(&task);
                }
            }
        }
        // # Clear task read deps
        // Clearing the existing vector might be faster than
        // replacing it with an empty one, since the latter
        // would trigger memory allocation.
        
        if let Some(read_deps) = self.read_deps_per_task.get_mut(&task) {
            read_deps.clear();
        }

        // # Write deps
        // Trigger changes to depending tasks
        if let Some(write_deps) = self.write_deps_per_task.get(&task) {
            for w_dep in write_deps {
                if let Some(waw_dependent_task) = self.writer_task_per_dep.get(&w_dep) {
                    if *waw_dependent_task != task {
                        let counter = self.dep_counter_per_task.get_mut(waw_dependent_task).unwrap();
                        *counter -= 1;
                    } else {
                        self.writer_task_per_dep.remove(&w_dep);
                    }
                }

                if let Some(raw_dependent_tasks) = self.reader_tasks_per_dep.get(&w_dep) {
                    for raw_dependent_task in raw_dependent_tasks {
                        let counter = self.dep_counter_per_task.get_mut(raw_dependent_task).unwrap();
                        *counter -= 1;
                    }
                }
            }
        }
    }

    pub fn pop_ready_task(&mut self) -> Option<usize> {
        self.ready_tasks.pop()
    }

    pub fn num_ready_tasks(&self) -> usize {
        self.ready_tasks.len()
    }

    pub fn clear(&mut self) {
        self.writer_task_per_dep.clear();
        self.reader_tasks_per_dep.clear();
        self.write_deps_per_task.clear();
        self.read_deps_per_task.clear();
        self.task_dependency_graph.clear();
        self.ready_tasks.clear();
        self.dep_counter_per_task.clear();
    }

    pub fn visualize_graph(&self, output_name: Option<String>) {
        visualize_graph(&self.task_dependency_graph.clone().into_graph(), None, output_name);
    }
}

static mut N6_DEP_GRAPH: Option<DepGraph> = None;
static mut TASK_COUNTER: usize = 0;
static mut INITIALIZED: bool = false;
static mut FINISHED_PARENT_TASK: bool = false;
const CYCLE_LENGTH: usize = 256;

#[no_mangle]
pub fn hello_partior() {
    println!("[Rust]: Hello from the other side!");
}

#[no_mangle]
pub fn cb_initialize_partior() {
    println!("[partior::cb_initialize_partior]");

    unsafe {
        N6_DEP_GRAPH = Some(DepGraph::new(true));
        INITIALIZED = true;
    }
}

#[no_mangle]
pub fn cb_add_task() {
    println!("[partior::cb_add_task]: Hello from the other side!");

    unsafe {
        if !INITIALIZED {
            N6_DEP_GRAPH = Some(DepGraph::new(true));
            INITIALIZED = true;

            // We do not add the first task, which
            // is merely the main parent task
        } else {
            N6_DEP_GRAPH.as_mut().unwrap().add_task(TASK_COUNTER);
        }
    }
}

#[no_mangle]
pub fn cb_remove_task() {
    println!("[partior::cb_remove_task]: Hello from the other side!");
}

#[no_mangle]
pub fn cb_add_dep(dep_type: u64, address: u64) {
    println!("[partior::cb_add_dep]: (dep_type, address) = ({}, {})", dep_type, address);
    unsafe {
        let dep_graph = N6_DEP_GRAPH.as_mut().unwrap();
        if dep_type == 1 {
            dep_graph.add_task_read_dep(TASK_COUNTER, address as usize);
        } else {
            println!("[partior::cb_add_dep]: Adding write dep = {}", address);
            dep_graph.add_task_write_dep(TASK_COUNTER, address as usize);
        }
    }
}

#[no_mangle]
pub fn cb_allocation_hint() {
    println!("[partior::cb_allocation_hint]: Hello from the other side!");
}

#[no_mangle]
pub fn cb_finish_adding_task() {
    println!("[partior::cb_allocation_hint]: Hello from the other side!");
    unsafe {
        if FINISHED_PARENT_TASK {
            let dep_graph = N6_DEP_GRAPH.as_mut().unwrap();
            dep_graph.finish_adding_task(TASK_COUNTER);

            TASK_COUNTER += 1;
            if TASK_COUNTER > 0 && (TASK_COUNTER % CYCLE_LENGTH) == 0 {
                N6_DEP_GRAPH.as_ref().unwrap().visualize_graph(Some(format!("{CYCLE_LENGTH}_tasks_up_to{TASK_COUNTER}").to_string()));
                dep_graph.clear();
            }
        } else {
            FINISHED_PARENT_TASK = true;
        }
    }
}
