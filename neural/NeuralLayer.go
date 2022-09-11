package neural

type NeuralLayer struct {
	// NeuronUnits in layer
	NeuronUnits []NeuronUnit
	// number of NeuronUnit in layer
	Length int
}

func PrepareLayer(numberOfNeuronsNeuralLayer int,
	numberOfNeuronsPreviousNeuralLayer int) (layer NeuralLayer) {
	layer = NeuralLayer{NeuronUnits: make([]NeuronUnit, numberOfNeuronsNeuralLayer),
		Length: numberOfNeuronsNeuralLayer}
	for i := 0; i < numberOfNeuronsNeuralLayer; i++ {
		RandomNeuronInit(&layer.NeuronUnits[i], numberOfNeuronsPreviousNeuralLayer)
	}
	return
}
