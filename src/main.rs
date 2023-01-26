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

const MAX_NUMBER_OF_PARTITIONS: usize = 8;

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
fn write_to_file(s: &String) {
    const TEMP_FILE_NAME: &str = "temp.dot";
    let mut f = File::create(TEMP_FILE_NAME).unwrap();
    //println!("{}", s);
    f.write(s.as_bytes()).unwrap();

    gen_graph_image(TEMP_FILE_NAME);
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
    fn internal_edge_count(&self) -> usize;
    fn calculate_max_internal_links(&self) -> usize;
}

impl CanCountInternalLinks for Graph<NodeInfo, usize, petgraph::Directed, usize> {
    fn internal_edge_count(&self) -> usize {
        let mut num_internal_links : usize = 0;

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

        //println!("[internal_edge_count]: num_internal_links = {}", num_internal_links);
        num_internal_links
    }

    fn calculate_max_internal_links(&self) -> usize {
        let mut items_per_partition = HashMap::new();

        for v in self.node_references() {
            let pid = v.weight().partition_id();

            let hash_count = items_per_partition.entry(pid).or_insert(0);
            *hash_count += 1;
        }

        let mut max_internal_links = 0;

        for n in items_per_partition.values() {
            max_internal_links += n * (n - 1) / 2;
        }

        //println!("[calculate_max_internal_links]: Partition size HashMap: {:?}", items_per_partition);
        //println!("[calculate_max_internal_links]: max_internal_links: {}", max_internal_links);

        max_internal_links
    }
}

// Implemented according to the definition found in https://doi.org/10.1371/journal.pone.0024195 .
// Better clusterings have a higher surprise value.
fn calculate_surprise(g: &Graph<NodeInfo, usize, petgraph::Directed, usize>) -> f64 {
    let num_nodes = g.node_count();

    let num_links : u64 = g.edge_count() as u64;
    let num_max_links : u64 = (num_nodes * (num_nodes - 1) / 2) as u64;

    let num_internal_links : u64 = g.internal_edge_count() as u64;
    let num_max_internal_links = g.calculate_max_internal_links() as u64;

    let top = min(num_links, num_max_internal_links);
    let mut surprise: f64 = 0.0;

    for j in num_internal_links..=top {
        surprise -= (binomial(num_max_internal_links, j)) * (binomial(num_max_links - num_max_internal_links, num_links - j));
    }
    surprise /= binomial(num_max_links, num_links);

    //println!("Graph surprise: {}", surprise);

    surprise
}

fn gen_random_digraph(acyclic: bool, max_num_nodes: usize, exact_num_nodes: usize) -> petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize> {
    let mut g = GraphMap::<NodeInfo, usize, petgraph::Directed>::new();

    let mut rng = rand::thread_rng();

    let num_nodes: usize;
    if exact_num_nodes == 0 {
        num_nodes = rng.gen_range(1..=max_num_nodes);
    } else {
        num_nodes = exact_num_nodes;
    }
    let max_num_edges = num_nodes * (num_nodes - 1) / 2;
    let num_edges: usize = rng.gen_range(0..=max_num_edges);

    println!("Number of nodes: {}", num_nodes);
    println!("Number of edges: {}", num_edges);

    for i in 0..num_nodes {
        g.add_node(NodeInfo{numerical_id: i, partition_id: 0});
    }

    for i in 0..num_edges {
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
            b = rng.gen_range(a+1..num_nodes);
        }

        g.add_edge(NodeInfo {numerical_id: a, partition_id: 0}, NodeInfo {numerical_id: b, partition_id: 0}, i);
    }
    
    let g : petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize> = g.into_graph();

    // The following is equivalent to 'return g;'
    g
}

fn gen_histogram(csv_path_with_suffix: &'static str, image_output_without_suffix: &'static str) {
    let _dot_proc_output = Command::new("src/histogram.py").arg(csv_path_with_suffix).arg(image_output_without_suffix).output().unwrap();
}

fn gen_graph_image(file_name: &'static str) {
    let _dot_proc_output = Command::new("dot").arg("-Tpng").arg(file_name).arg("-o").arg("output.png").output().unwrap();
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

#[allow(dead_code)]
fn get_number_of_partitions(g: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>) -> usize {
    let mut items_per_partition = HashMap::new();

    for v in g.node_references() {
        let pid = v.weight().partition_id();
        let _ = items_per_partition.entry(pid).or_insert(0);
    }

    items_per_partition.len()
}

fn visualize_graph(g: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>) {
    let null_out = |_, _| "".to_string();

    //let num_partitions = get_number_of_partitions(g);

    fn node_attr_generator<P: petgraph::visit::NodeRef>(_: &&Graph<NodeInfo, usize, petgraph::Directed, usize>, node_ref: P) -> String where <P as petgraph::visit::NodeRef>::Weight: fmt::Debug + HasPartition {
        let w = node_ref.weight();
        //let c = COLORS[w.partition_id()];
        let c = get_equally_hue_spaced_hsv_string(w.partition_id(), MAX_NUMBER_OF_PARTITIONS);
        format!("style=filled, color=\"{}\", fillcolor=\"{}\"", c, c).to_string()
    }

    let dot_dump = format!("{:?}", Dot::with_attr_getters(&g, &[Config::EdgeNoLabel], &null_out, &node_attr_generator));
    
    let _ = write_to_file(&dot_dump);
    calculate_surprise(&g);
}

fn set_random_partitions_and_visualize_graph(graph: &mut petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, max_partitions: usize) -> &Graph<NodeInfo, usize, petgraph::Directed, usize>{
    set_random_partitions(graph, max_partitions);
    visualize_graph(&graph);
    graph
}

fn get_equally_hue_spaced_hsv_string(index: usize, num_items: usize) -> String {
    let h: f64 = (index as f64) / (num_items as f64);
    let res = format!("{} 0.600 1.00", h);
    //println!("{}", res);
    res
}

fn evaluate_multiple_random_clusterings(original_graph: &petgraph::Graph<NodeInfo, usize, petgraph::Directed, usize>, max_partitions: usize, num_iterations: usize) {
    const CSV_PATH_WITH_SUFFIX : &str = "surprise.csv";
    const HIST_PATH_WITHOUT_SUFFIX : &str = "surprise_hist";
    
    let mut g = original_graph.clone();
    let mut wtr = Writer::from_path(CSV_PATH_WITH_SUFFIX).unwrap();

    // Header
    wtr.write_record(&["surprise"]).unwrap();

    let mut best_surprise_so_far: f64 = -1.0;
    for i in 0..num_iterations {
        set_random_partitions(&mut g, max_partitions);
        let s = calculate_surprise(&g);
        //println!("[evaluate_multiple_random_clusterings]: {}", i);
        if s > best_surprise_so_far {
            visualize_graph(&g);
            best_surprise_so_far = s;
            println!("[evaluate_multiple_random_clusterings]: (iteration, best_surprise_so_far): ({}, {})", i, s);
        }

        wtr.write_record(&[s.to_string()]).unwrap();
    }

    wtr.flush().unwrap();
    gen_histogram(CSV_PATH_WITH_SUFFIX, HIST_PATH_WITHOUT_SUFFIX);
}

#[allow(dead_code)]
fn gen_sample_graph_image() {
    //let g = gen_sample_graph();
    let mut g = gen_random_digraph(true, 20, 64);
    let new_graph = set_random_partitions_and_visualize_graph(&mut g, MAX_NUMBER_OF_PARTITIONS);
    visualize_graph(&new_graph);
}

fn test_histogram_01() {
    let g = gen_random_digraph(true, 20, 32);
    evaluate_multiple_random_clusterings(&g, MAX_NUMBER_OF_PARTITIONS, 3000000);
}

fn main() {
    //gen_sample_graph_image();
    test_histogram_01();
}
