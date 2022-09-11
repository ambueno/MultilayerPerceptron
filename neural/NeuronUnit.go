package neural

type NeuronUnit struct {
	// the way each dimensions of the pattern is modulated
	Weights []float64
	// NeuronUnit natural propensity to spread signal
	Bias float64
	// learning rate of neuron
	LearningRate float64
	// the desired value when the input pattern is loaded into network
	Value float64
	//  maintains error during execution of training algorithm
	Delta float64
}

func RandomNeuronInit(neuron *NeuronUnit, dimension int) {

}
