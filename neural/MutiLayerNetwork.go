package neural

type MultiLayerNetwork struct {
	// layer of neurons
	NeuralLayers []NeuralLayer
	// learning rate of neuron
	LearningRate float64
	// transfer function
	TransferFunction transferFunction
	// transfer function derivative
	TransferFunctionDerivative transferFunction
}
