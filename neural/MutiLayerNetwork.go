package neural

import (
	log "github.com/sirupsen/logrus"
	"math"
)

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

// Execute a multi layer Perceptron neural network.
// [multiLayerPerceptron:MultiLayerNetwork] multilayer perceptron network pointer, [input:Pattern] input value
// It returns output values by network
func Execute(multiLayerPerceptron *MultiLayerNetwork, input *Pattern, options ...int) (output []float64) {
	output = make([]float64, multiLayerPerceptron.NeuralLayers[len(multiLayerPerceptron.NeuralLayers)-1].Length)

	for i := 0; i < len(input.Features); i++ {
		multiLayerPerceptron.NeuralLayers[0].NeuronUnits[i].Value = input.Features[i]
	}

	// todo: reduce time complexity
	for i := 0; i < len(multiLayerPerceptron.NeuralLayers); i++ {
		for j := 0; j < multiLayerPerceptron.NeuralLayers[i].Length; j++ {
			newValue := 0.0
			for k := 0; k < multiLayerPerceptron.NeuralLayers[i-1].Length; k++ {
				newValue += multiLayerPerceptron.NeuralLayers[i].NeuronUnits[j].Weights[k] * multiLayerPerceptron.NeuralLayers[i-1].NeuronUnits[k-1].Value
				log.WithFields(log.Fields{
					"level":                 "debug",
					"msg":                   "multilayer perceptron execution",
					"len(mlp.NeuralLayers)": len(multiLayerPerceptron.NeuralLayers),
					"layer:  ":              i,
					"neuron: ":              j,
					"previous neuron: ":     k,
				}).Debug("Compute output propagation.")
			}
			newValue += multiLayerPerceptron.NeuralLayers[i].NeuronUnits[j].Bias
			multiLayerPerceptron.NeuralLayers[i].NeuronUnits[j].Value = multiLayerPerceptron.TransferFunction(newValue)
			log.WithFields(log.Fields{
				"level":                 "debug",
				"msg":                   "setup new neuron output value after transfer function application",
				"len(mlp.NeuralLayers)": len(multiLayerPerceptron.NeuralLayers),
				"layer:  ":              i,
				"neuron: ":              j,
				"outputvalue":           multiLayerPerceptron.NeuralLayers[i].NeuronUnits[j].Value,
			}).Debug("Setup new neuron output value after transfer function application.")
		}
	}

	for i := 0; i < multiLayerPerceptron.NeuralLayers[len(multiLayerPerceptron.NeuralLayers)-1].Length; i++ {
		output[i] = multiLayerPerceptron.NeuralLayers[len(multiLayerPerceptron.NeuralLayers)-1].NeuronUnits[i].Value
	}

	return output
}

func BackPropagate(multiLayerPerceptron *MultiLayerNetwork, input *Pattern, expectedOutput []float64, options ...int) (deltaError float64) {
	var newExpectedOutput []float64
	if len(options) == 1 {
		newExpectedOutput = Execute(multiLayerPerceptron, input, options[0])
	} else {
		newExpectedOutput = Execute(multiLayerPerceptron, input, options[0])
	}

	errorValue := 0.0
	for i := 0; i < multiLayerPerceptron.NeuralLayers[len(multiLayerPerceptron.NeuralLayers)-1].Length; i++ {
		errorValue = expectedOutput[i] - newExpectedOutput[i]
		multiLayerPerceptron.NeuralLayers[len(multiLayerPerceptron.NeuralLayers)-1].NeuronUnits[i].Delta = errorValue * multiLayerPerceptron.TransferFunctionDerivative(newExpectedOutput[i])
	}

	// todo: reduce time complexity
	for i := len(multiLayerPerceptron.NeuralLayers) - 2; i >= 0; i-- {
		for j := 0; j < multiLayerPerceptron.NeuralLayers[i].Length; j++ {
			errorValue = 0.0
			for k := 0; k < multiLayerPerceptron.NeuralLayers[i+1].Length; k++ {
				errorValue += multiLayerPerceptron.NeuralLayers[i+1].NeuronUnits[k].Delta * multiLayerPerceptron.NeuralLayers[i+1].NeuronUnits[k].Weights[j]
			}
			multiLayerPerceptron.NeuralLayers[i].NeuronUnits[j].Delta = errorValue * multiLayerPerceptron.TransferFunctionDerivative(multiLayerPerceptron.NeuralLayers[i].NeuronUnits[j].Value)
		}
		for j := 0; j < multiLayerPerceptron.NeuralLayers[i+1].Length; j++ {
			for k := 0; k < multiLayerPerceptron.NeuralLayers[i].Length; k++ {
				multiLayerPerceptron.NeuralLayers[i+1].NeuronUnits[j].Weights[k] += multiLayerPerceptron.LearningRate * multiLayerPerceptron.NeuralLayers[i+1].NeuronUnits[j].Delta * multiLayerPerceptron.NeuralLayers[i].NeuronUnits[k].Value
			}
			multiLayerPerceptron.NeuralLayers[i+1].NeuronUnits[j].Bias += multiLayerPerceptron.LearningRate * multiLayerPerceptron.NeuralLayers[i+1].NeuronUnits[j].Delta
		}
	}
	for i := 0; i < len(expectedOutput); i++ {
		deltaError += math.Abs(newExpectedOutput[i] - expectedOutput[i])
	}
	deltaError = deltaError / float64(len(expectedOutput))
	return
}
