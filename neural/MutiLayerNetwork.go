package neural

import (
	"MultilayerPerceptron/util"
	log "github.com/sirupsen/logrus"
	"math"
	"math/rand"
	"os"
	_ "os"
	"time"
)

func init() {
	// Output to stdout instead of the default stderr
	log.SetOutput(os.Stdout)
	// Only log the warning severity or above.
	log.SetLevel(log.InfoLevel)
}

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

// PrepareElmanNet create a recurrent neural network.
// [inputLayer:int]
// [hiddenLayer:int]
// [outputLayer:int]
// [learningRate:int] is the learning rate of neural network
// [tf:transferFunction] is a transfer function
// [tfd:transferFunction] the respective transfer function derivative
func PrepareElmanNet(inputLayer int, hiddenLayer int, outputLayer int, learningRate float64, tf transferFunction, tfd transferFunction) (rnn MultiLayerNetwork) {
	rnn = PrepareMLPNet([]int{inputLayer, hiddenLayer, outputLayer}, learningRate, tf, tfd)

	log.WithFields(log.Fields{
		"level":          "info",
		"msg":            "recurrent neural network init completed",
		"inputLayer":     inputLayer,
		"hiddenLayer":    hiddenLayer,
		"outputLayer":    outputLayer,
		"learningRate: ": rnn.LearningRate,
	}).Info("Complete RNN init.")
	return
}

// Execute a multi layer Perceptron neural network.
// [multiLayerPerceptron:MultiLayerNetwork] multilayer perceptron network pointer,
// [input:Pattern] input value
// It returns output values by network
func Execute(multiLayerPerceptron *MultiLayerNetwork, input *Pattern, options ...int) (output []float64) {
	output = make([]float64, multiLayerPerceptron.NeuralLayers[len(multiLayerPerceptron.NeuralLayers)-1].Length)

	for i := 0; i < len(input.Features); i++ {
		multiLayerPerceptron.NeuralLayers[0].NeuronUnits[i].Value = input.Features[i]
	}

	for i := len(input.Features); i < multiLayerPerceptron.NeuralLayers[0].Length; i++ {
		multiLayerPerceptron.NeuralLayers[0].NeuronUnits[i].Value = 0.5
	}

	// todo: reduce time complexity
	for i := 1; i < len(multiLayerPerceptron.NeuralLayers); i++ {
		for j := 0; j < multiLayerPerceptron.NeuralLayers[i].Length; j++ {
			newValue := 0.0
			for k := 0; k < multiLayerPerceptron.NeuralLayers[i-1].Length; k++ {
				newValue += multiLayerPerceptron.NeuralLayers[i].NeuronUnits[j].Weights[k] * multiLayerPerceptron.NeuralLayers[i-1].NeuronUnits[k].Value
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
			if i == 1 && len(options) > 0 && options[0] == 1 {
				for j := len(input.Features); j < multiLayerPerceptron.NeuralLayers[0].Length; j++ {
					log.WithFields(log.Fields{
						"level":                               "debug",
						"len z":                               j,
						"s.Features":                          input.Features,
						"len(s.Features)":                     len(input.Features),
						"len mlp.NeuralLayers[0].NeuronUnits": len(multiLayerPerceptron.NeuralLayers[0].NeuronUnits),
						"len mlp.NeuralLayers[k].NeuronUnits": len(multiLayerPerceptron.NeuralLayers[i].NeuronUnits),
					}).Debug("Save output of hidden layer to context.")

					multiLayerPerceptron.NeuralLayers[0].NeuronUnits[j].Value = multiLayerPerceptron.NeuralLayers[i].NeuronUnits[j-len(input.Features)].Value

				}

			}
			log.WithFields(log.Fields{
				"level":                 "debug",
				"msg":                   "setup new neuron output value after transfer function application",
				"len(mlp.NeuralLayers)": len(multiLayerPerceptron.NeuralLayers),
				"layer:  ":              i,
				"neuron: ":              j,
				"output-value":          multiLayerPerceptron.NeuralLayers[i].NeuronUnits[j].Value,
			}).Debug("Setup new neuron output value after transfer function application.")
		}
	}

	for i := 0; i < multiLayerPerceptron.NeuralLayers[len(multiLayerPerceptron.NeuralLayers)-1].Length; i++ {
		output[i] = multiLayerPerceptron.NeuralLayers[len(multiLayerPerceptron.NeuralLayers)-1].NeuronUnits[i].Value
	}

	return output
}

// BackPropagate BackPropagation algorithm.
// [multiLayerPerceptron:MultiLayerNetwork] input value
// [input:Pattern] input value (scaled between 0 and 1)
// [expectedOutput:[]float64] expected output value (scaled between 0 and 1)
// return [deltaError:float64] delta error between generated output and expected output
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

func MLPTrain(multiLayerPerceptron *MultiLayerNetwork, patterns []Pattern, mapped []string, epochs int) {
	epoch := 0
	output := make([]float64, len(mapped))
	for {
		for _, pattern := range patterns {
			for io := range output {
				output[io] = 0.0
			}
			output[int(pattern.SingleExpectation)] = 1.0
			BackPropagate(multiLayerPerceptron, &pattern, output)
		}

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "validation",
			"method": "MLPTrain",
			"epoch":  epoch,
		}).Debug("Training epoch completed.")

		if epoch > epochs {
			break
		}
		epoch++
	}
}

// ElmanTrain train a mlp MultiLayerNetwork with BackPropagation algorithm for assisted learning.
func ElmanTrain(mlp *MultiLayerNetwork, patterns []Pattern, epochs int) {
	epoch := 0
	for {
		rand.Seed(time.Now().UTC().UnixNano())
		pIR := rand.Intn(len(patterns))
		for pI, pattern := range patterns {
			BackPropagate(mlp, &pattern, pattern.MultipleExpectation, 1)
			if epoch%100 == 0 && pI == pIR {
				oOut := Execute(mlp, &pattern, 1)
				for oOutI, oOutV := range oOut {
					oOut[oOutI] = util.Round(oOutV, .5, 0)
				}
				log.WithFields(log.Fields{
					"SUM": "  ==========================",
				}).Info()
				log.WithFields(log.Fields{
					"a_n_1": util.ConvertBinToInt(pattern.Features[0:(len(pattern.Features) / 2)]),
					"a_n_2": pattern.Features[0:(len(pattern.Features) / 2)],
				}).Info()
				log.WithFields(log.Fields{
					"b_n_1": util.ConvertBinToInt(pattern.Features[(len(pattern.Features) / 2):]),
					"b_n_2": pattern.Features[(len(pattern.Features) / 2):],
				}).Info()
				log.WithFields(log.Fields{
					"sum_1": util.ConvertBinToInt(pattern.MultipleExpectation),
					"sum_2": pattern.MultipleExpectation,
				}).Info()
				log.WithFields(log.Fields{
					"sum_1": util.ConvertBinToInt(oOut),
					"sum_2": oOut,
				}).Info()
				log.WithFields(log.Fields{
					"END": "  ==========================",
				}).Info()
			}
		}

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "validation",
			"method": "ElmanTrain",
			"epoch":  epoch,
		}).Debug("Training epoch completed.")

		if epoch > epochs {
			break
		}
		epoch++
	}
}
