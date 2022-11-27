package neural

import log "github.com/sirupsen/logrus"

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

// PrepareMLPNet create a multi layer Perceptron neural network.
// [layer:[]int] is an int array with layers neurons number [input, ..., output]
// [learningRate:int] is the learning rate of neural network
// [tf:transferFunction] is a transfer function
// [tfd:transferFunction] the respective transfer function derivative
func PrepareMLPNet(layer []int, learningRate float64, tf transferFunction, tfd transferFunction) (multiLayerPerceptron MultiLayerNetwork) {
	multiLayerPerceptron.LearningRate = learningRate
	multiLayerPerceptron.TransferFunction = tf
	multiLayerPerceptron.TransferFunctionDerivative = tfd

	multiLayerPerceptron.NeuralLayers = make([]NeuralLayer, len(layer))

	for iLayer, jLayer := range layer {
		if iLayer != 0 {
			multiLayerPerceptron.NeuralLayers[iLayer] = PrepareLayer(jLayer, layer[iLayer-1])
		} else {
			multiLayerPerceptron.NeuralLayers[iLayer] = PrepareLayer(jLayer, 0)
		}
	}
	log.WithFields(log.Fields{
		"level":          "info",
		"msg":            "multilayer perceptron init completed",
		"layers":         len(multiLayerPerceptron.NeuralLayers),
		"learningRate: ": multiLayerPerceptron.LearningRate,
	}).Info("Complete Multilayer Perceptron init.")

	return
}

func Execute(multiLayerPerceptron *MultiLayerNetwork, input *Pattern, options ...int) (output []float64) {

}
