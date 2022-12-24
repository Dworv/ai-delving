use ecosystem::NeuralNetwork;

fn main() {
    let mut nn: NeuralNetwork = NeuralNetwork::new(2, 1, 1, 1);
    println!("{:?}", nn);
    println!("{:?}", nn.run(vec![1., -1.]));
    println!("{:?}", nn);
    for _ in 0..10 {
        nn.mutate_add_neuron();
        nn.mutate_add_connection()
    }
    println!("{:?}", nn);
    println!("{:?}", nn.run(vec![1., -1.]))
}
