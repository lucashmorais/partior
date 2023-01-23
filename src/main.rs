use petgraph::graph::{NodeIndex, UnGraph, Graph};
use petgraph::graphmap::{GraphMap};
use petgraph::algo::{dijkstra, min_spanning_tree};
use petgraph::data::FromElements;
use petgraph::dot::{Dot, Config};
use std::fs::File;
use std::io::prelude::*;
use std::process::Command;
use rand::Rng;

#[derive(Clone,Copy,Eq,PartialEq,PartialOrd,Hash,Debug,Ord)]
struct NodeInfo {
    numerical_id: usize,
    partition_id: usize
}

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
    println!("{}", s);
    f.write(s.as_bytes()).unwrap();

    gen_graph_image(TEMP_FILE_NAME);
}

// Create an undirected graph with `i32` nodes and edges with `()` associated data.
fn gen_sample_graph() -> UnGraph<i32, ()> {
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

// TODO: Add an option for ensuring that the graph is acyclic
fn gen_random_digraph(acyclic: bool) -> GraphMap<NodeInfo, usize, petgraph::Directed> {
    let mut g = GraphMap::<NodeInfo, usize, petgraph::Directed>::new();

    let mut rng = rand::thread_rng();

    let num_nodes: usize = rng.gen_range(1..=20);
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

    // The following is equivalent to 'return g;'
    g
}

fn gen_graph_image(file_name: &'static str) {
    let _dot_proc_output = Command::new("dot").arg("-Tpng").arg(file_name).arg("-o").arg("output.png").output().unwrap();
}

fn gen_sample_graph_image() {
    //let g = gen_sample_graph();
    let g = gen_random_digraph(true);
    let null_out = |_, _| "".to_string();

    // Output the tree to `graphviz` `DOT` format
    //
    // We use a '{:?}' modifier here because the Dot class implements the dmt::Debug
    // trait, but not the fmt::Display trait that is required by a mere '{}'
    // See https://github.com/rust-lang/rfcs/blob/master/text/0565-show-string-guidelines.md
    // for details

    let mut partition: Vec<usize> = vec![];

    let mut gen_random_partitions = |graph: &GraphMap<NodeInfo, usize, petgraph::Directed>| {
        let mut rng = rand::thread_rng();

        let nodes = graph.nodes();
        let num_nodes = graph.node_count();

        for n in nodes {
            let partition_idx = rng.gen_range(0..num_nodes);
            partition.push(partition_idx);
            println!("(node_idx, partition_idx) = ({:?}, {})", n, partition_idx);
        }
    };

    gen_random_partitions(&g);

    fn node_attr_generator<P: petgraph::visit::NodeRef>(_: &GraphMap<NodeInfo, usize, petgraph::Directed>, node_ref: P) -> String where <P as petgraph::visit::NodeRef>::Weight: std::fmt::Debug {
        println!("Node id: {:?}", node_ref.weight());
        //println!("{}", partition);
        "color=\"green\"".to_string()
    }

    let dot_dump = format!("{:?}", Dot::with_attr_getters(&g, &[Config::EdgeNoLabel], &null_out, &node_attr_generator));
    
    let _ = write_to_file(&dot_dump);
}

fn main() {
    //let output_str: String = String::from_utf8()?;

    // Here we use "unwrap()" instead of "?" to implement void
    // error handling because "?" includes a return statement
    // that would require this function to return a Result<(), Error>
    //
    // We could have used 'expect("some_error_msg")' as well
    gen_sample_graph_image();
}
