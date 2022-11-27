package neural

import (
	"MultilayerPerceptron/util"
	log "github.com/sirupsen/logrus"
	"math/rand"
	"os"
)

const (
	ScalingFactor = 0.0000000000001
)

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

func init() {
	log.SetOutput(os.Stdout)
	log.SetLevel(log.InfoLevel)
}

// RandomNeuronInit initialize neuron weight, bias and learning rate using NormFloat64 random value.
func RandomNeuronInit(neuron *NeuronUnit, dim int) {
	neuron.Weights = make([]float64, dim)
	for index := range neuron.Weights {
		neuron.Weights[index] = rand.NormFloat64() * ScalingFactor
	}

	neuron.Bias = rand.NormFloat64() * ScalingFactor
	neuron.LearningRate = rand.NormFloat64() * ScalingFactor
	neuron.Value = rand.NormFloat64() * ScalingFactor
	neuron.Delta = rand.NormFloat64() * ScalingFactor

	log.WithFields(log.Fields{
		"level":   "debug",
		"place":   "neuron",
		"func":    "RandomNeuronInit",
		"msg":     "random neuron weights init",
		"weights": neuron.Weights,
	}).Debug()
}

// UpdateWeights performs update in neuron weights with respect to passed pattern.
// It returns error of prediction before and after updating weights.
func UpdateWeights(neuron *NeuronUnit, pattern *Pattern) (float64, float64) {
	var predictedValue, prevError, postError = Predict(neuron, pattern), 0.0, 0.0
	prevError = pattern.SingleExpectation - predictedValue
	neuron.Bias = neuron.Bias + neuron.LearningRate*prevError

	for index := range neuron.Weights {
		neuron.Weights[index] = neuron.Weights[index] + neuron.LearningRate*prevError*pattern.Features[index]
	}

	predictedValue = Predict(neuron, pattern)
	postError = pattern.SingleExpectation - predictedValue

	log.WithFields(log.Fields{
		"level":   "debug",
		"place":   "neuron",
		"func":    "UpdateWeights",
		"msg":     "updating weights of neuron",
		"weights": neuron.Weights,
	}).Debug()

	return prevError, postError
}

// TrainNeuron trains a passed neuron with patterns passed, for specified number of epoch.
// If init is 0, leaves weights unchanged before training.
// If init is 1, reset weights and bias of neuron before training.
func TrainNeuron(neuron *NeuronUnit, patterns []Pattern, epochs int, init int) {
	if init == 1 {
		neuron.Weights = make([]float64, len(patterns[0].Features))
		neuron.Bias = 0.0
	}

	var epoch = 0
	var squaredPrevError, squaredPostError = 0.0, 0.0
	for epoch < epochs {
		for _, pattern := range patterns {
			prevError, postError := UpdateWeights(neuron, &pattern)
			squaredPrevError = squaredPrevError + (prevError * prevError)
			squaredPostError = squaredPostError + (postError * postError)
		}

		log.WithFields(log.Fields{
			"level":            "debug",
			"place":            "error evolution in epoch",
			"method":           "TrainNeuron",
			"msg":              "epoch and squared errors reached before and after updating weights",
			"epochReached":     epoch + 1,
			"squaredErrorPrev": squaredPrevError,
			"squaredErrorPost": squaredPostError,
		}).Debug()

		epoch++
	}
}

// Predict performs a neuron prediction to passed pattern.
// It returns a float64 binary predicted value.
func Predict(neuron *NeuronUnit, pattern *Pattern) float64 {
	if util.ScalarProduct(neuron.Weights, pattern.Features)+neuron.Bias < 0.0 {
		return 0.0
	}
	return 1.0
}

// Accuracy calculate percentage of equal values between two float64 based slices.
// It returns int number and a float64 percentage value of corrected values.
func Accuracy(actual []float64, predicted []float64) (int, float64) {
	if len(actual) != len(predicted) {
		log.WithFields(log.Fields{
			"level":        "error",
			"place":        "neuron",
			"method":       "Accuracy",
			"msg":          "accuracy between actual and predicted slices of values",
			"actualLen":    len(actual),
			"predictedLen": len(predicted),
		}).Error("Failed to compute accuracy between actual values and predictions: different length.")
		return -1, -1.0
	}

	var correct = 0

	for index, value := range actual {
		if value == predicted[index] {
			correct++
		}
	}

	return correct, float64(correct) / float64(len(actual)) * 100.0
}
