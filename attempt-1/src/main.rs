use ecosystem::NeuralNetwork;

fn main() {
    let mut champion = (0.0, NeuralNetwork::new(2, 1, 2, 5));

    for _ in 0..20 {
        for _ in 0..100 {
            let mut child = champion.1.child();
            let fitness = xor_fitness(&mut child);
            if fitness > champion.0 {
                champion = (fitness, child);
            }
        }
    }

    println!("champion was {:?}, fitness was {:?}", champion.1, champion.0);
    println!("champion says 3^2 = {}", champion.1.run(vec![3., 2.])[0])
}

fn xor_fitness(network: &mut NeuralNetwork) -> f64 {
    let mut fitness = 0.0;
    for i in 1..=10 {
        for j in 1..=10 {
            let xor = (i^j) as f64;
            let guess = network.run(vec![i as f64, j as f64])[0];
            let score = if xor - guess == 0. {
                1.
            } else {
                1. - 1. / (xor-guess).abs()
            };
            println!("xor: {}, guess: {}, score: {}", xor, guess, score);
            fitness += score / 40.0f64.powf(2.);
        }
    }
    dbg!(fitness)
}