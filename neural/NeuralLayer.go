package neural

import (
	log "github.com/sirupsen/logrus"
	_ "os"
)

type NeuralLayer struct {
	// NeuronUnits in layer
	NeuronUnits []NeuronUnit
	// number of NeuronUnit in layer
	Length int
}

// PrepareLayer creates a NeuralLayer with
// numberOfNeuronsNeuralLayer NeuronUnits inside
// It returns a NeuralLayer
func PrepareLayer(numberOfNeuronsNeuralLayer int,
	numberOfNeuronsPreviousNeuralLayer int) (layer NeuralLayer) {
	layer = NeuralLayer{NeuronUnits: make([]NeuronUnit, numberOfNeuronsNeuralLayer),
		Length: numberOfNeuronsNeuralLayer}
	for i := 0; i < numberOfNeuronsNeuralLayer; i++ {
		RandomNeuronInit(&layer.NeuronUnits[i], numberOfNeuronsPreviousNeuralLayer)
	}
	log.WithFields(log.Fields{
		"level":               "info",
		"msg":                 "multilayer perceptron init completed",
		"neurons":             len(layer.NeuronUnits),
		"lengthPreviousLayer": layer.Length,
	}).Info("Complete NeuralLayer init.")
	return
}
