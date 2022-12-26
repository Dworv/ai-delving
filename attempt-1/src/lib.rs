use rand::{Rng, thread_rng};

#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    layers: Vec<Vec<Neuron>>,
    outputs: Vec<Output>,
    mutations: usize
}

impl NeuralNetwork {
    pub fn new(inputs: usize, outputs: usize, max_layers: usize, mutations: usize) -> Self {
        let mut layers = Vec::with_capacity(max_layers + 1);
        layers.push(vec![Neuron::new(); inputs]);
        layers.append(&mut vec![Vec::new(); max_layers]);
        Self {
            layers,
            outputs: vec![Output::new(); outputs],
            mutations
        }
    }

    fn get_mut_neuron(&mut self, layer: usize, index: usize) -> Option<&mut Neuron> {
        self.layers.get_mut(layer).unwrap().get_mut(index)
    }

    fn get_mut_output(&mut self, index: usize) -> Option<&mut Output> {
        self.outputs.get_mut(index)
    }

    pub fn mutate_add_connection(&mut self) {
        let num_layers = self.layers.len();
        let rand_layer = thread_rng().gen_range(0..num_layers);
        let layer = &self.layers[rand_layer];
        let num_neurons = layer.len();
        if num_neurons > 0 {
            let rand_neuron = thread_rng().gen_range(0..num_neurons);
            let neuron = &layer[rand_neuron];
            if thread_rng().gen() {
                // connection to output
                let num_outputs = self.outputs.len();
                let rand_output = thread_rng().gen_range(0..num_outputs);
                if matches!(neuron.connections.iter().filter(|x| {
                    if let Target::Output(index) = x.target {
                        index == rand_output
                    } else { false }
                }).next(), None) {
                    self.layers[rand_layer][rand_neuron].connections.push(Connection::new(Target::Output(rand_output)))
                }
            } else if rand_layer < num_layers - 1 {
                // connection to neuron
                let rand_further_layer = thread_rng().gen_range((rand_layer + 1)..num_layers);
                let num_further_neurons = self.layers[rand_further_layer].len();
                if num_further_neurons > 0 {
                    let rand_further_neuron = thread_rng().gen_range(0..num_further_neurons);
                    if matches!(
                        neuron.connections.iter().filter(|x| {
                            if let Target::Neuron(l, i) = x.target {
                                l == rand_further_layer && i == rand_further_neuron
                            } else { false }
                        }).next(),
                        None   
                    ) {
                        self.layers[rand_layer][rand_neuron].connections.push(Connection::new(Target::Neuron(rand_further_layer, rand_further_neuron)))
                    }
                } 
            }
        }
    }

    pub fn mutate_add_neuron(&mut self) {
        if self.layers.len() > 1 {
            let num_inputs = self.layers[0].len();
            let rand_layer = thread_rng().gen_range(1..self.layers.len());
            let layer = &mut self.layers[rand_layer]; 
            if layer.len() < num_inputs {
                layer.push(Neuron::new());
            }
        }      
    }

    pub fn mutate_shift_weight(&mut self) {
        let num_layers = self.layers.len();
        let rand_layer = thread_rng().gen_range(0..num_layers);
        let layer = &mut self.layers[rand_layer];
        let num_neurons = layer.len();
        if num_neurons > 0 {
            let rand_neuron = thread_rng().gen_range(0..num_neurons);
            let neuron = &mut layer[rand_neuron];
            if neuron.connections.len() > 0 {
                let rand_conn = thread_rng().gen_range(0..neuron.connections.len());
                neuron.connections[rand_conn].weight += thread_rng().gen_range(-0.1..0.1);
            }
        }
    }

    pub fn child(&self) -> NeuralNetwork {
        let mut child = self.clone();
        for _ in 0..child.mutations {
            let rng: u8 = thread_rng().gen();
            match rng {
                0..=7 => child.mutate_add_neuron(),
                8..=71 => child.mutate_add_connection(),
                72.. => child.mutate_shift_weight()
            }
        }
        child
    }

    pub fn run(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0].len() {
            panic!("incorrect amount of inputs");
        }
        for (i, value) in inputs.iter().enumerate() {
            self.layers[0][i].recieve_input(*value);
        }
        for i in 0..self.layers.len() {
            for j in 0..self.layers[i].len() {
                let sum = self.layers[i][j].sum;
                for k in 0..self.layers[i][j].connections.len() {
                    let conn = self.layers[i][j].connections[k].clone();
                    match conn.target {
                        Target::Neuron(layer, index) => {
                            self.get_mut_neuron(layer, index).unwrap().recieve_input(sum * conn.weight)
                        },
                        Target::Output(index) => {
                            self.get_mut_output(index).unwrap().recieve_input(sum * conn.weight)
                        }
                    }
                }
                self.layers[i][j].sum = 0.0
            }
        }
        let mut outputs = vec![];
        for output in &mut self.outputs {
            outputs.push(output.sum);
            output.reset();
        }
        outputs
    }
}

#[derive(Clone, Debug)]
struct Neuron {
    connections: Vec<Connection>,
    sum: f64
}

impl Neuron {
    fn new() -> Self {
        Self {
            connections: vec![],
            sum: 0.,
        }
    }

    fn recieve_input(&mut self, input: f64) {
        self.sum += input;
    }
}

#[derive(Clone, Debug)]
struct Output {
    sum: f64
}

impl Output {
    fn new() -> Self {
        Self {
            sum: 0.
        }
    }

    fn recieve_input(&mut self, input: f64) {
        self.sum += input;
    }

    fn reset(&mut self) {
        self.sum = 0.
    }
}

#[derive(Clone, Debug)]
struct Connection {
    target: Target,
    weight: f64
}

impl Connection {
    fn new(target: Target) -> Self {
        Self {
            target,
            weight: thread_rng().gen_range((-0.1)..0.1)
        }
    }
}

#[derive(Clone, Debug)]
enum Target {
    Neuron(usize, usize),
    Output(usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simple_nn_run() {
        let mut nn = NeuralNetwork {
            layers: vec![vec![
                Neuron {
                    connections: vec![Connection { target: Target::Output(0), weight: 2.0 }],
                    sum: 0.
                },
            ]], 
            mutations: 1,
            outputs: vec![
                Output {
                    sum: 0.,

                }
            ]
        };
        let res = nn.run(vec![1.]);
        assert_eq!(res[0], 2.);
    }

    #[test]
    fn layered_nn_run() {
        let mut nn = NeuralNetwork {
            layers: vec![
                vec![
                    Neuron {
                        connections: vec![Connection { target: Target::Neuron(1, 0), weight: 2.0 }],
                        sum: 0.
                    },
                ],
                vec![
                    Neuron {
                        connections: vec![Connection { target: Target::Output(0), weight: 3.0 }],
                        sum: 0.
                    }
                ]
            ], 
            mutations: 1,
            outputs: vec![
                Output {
                    sum: 0.,

                }
            ]
        };
        let res = nn.run(vec![1.]);
        assert_eq!(res[0], 6.);
    }
}