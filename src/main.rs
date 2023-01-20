use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::algo::{dijkstra, min_spanning_tree};
use petgraph::data::FromElements;
use petgraph::dot::{Dot, Config};
use std::fs::File;
use std::io::prelude::*;
use std::process::Command;

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

fn gen_random_graph() {
}

fn gen_graph_image(file_name: &'static str) {
    let _dot_proc_output = Command::new("dot").arg("-Tpng").arg(file_name).arg("-o").arg("test.png").output().unwrap();
}

fn gen_sample_graph_image() {
    let g = gen_sample_graph();


    // Output the tree to `graphviz` `DOT` format
    //
    // We use a '{:?}' modifier here because the Dot class implements the dmt::Debug
    // trait, but not the fmt::Display trait that is required by a mere '{}'
    // See https://github.com/rust-lang/rfcs/blob/master/text/0565-show-string-guidelines.md
    // for details
    let dot_dump = format!("{:?}", Dot::with_config(&g, &[Config::EdgeNoLabel]));
    
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
