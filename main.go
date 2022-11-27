package main

import (
	"MultilayerPerceptron/neural"
	"MultilayerPerceptron/util"
	"MultilayerPerceptron/validation"
	log "github.com/sirupsen/logrus"
	"os"
)

func init() {
	log.SetOutput(os.Stdout)
	log.SetLevel(log.InfoLevel)
}

func main() {
	if true {
		log.WithFields(log.Fields{
			"level": "info",
			"place": "main",
			"msg":   "single layer perceptron train and test over sonar dataset",
		}).Info("Compute single layer perceptron on sonar data set (binary classification problem)")

		var filePath = "./resources/sonar.all_data.csv"
		var percentage = 0.67
		var shuffle = 1
		var bias = 0.0
		var learningRate = 0.01
		var epochs = 500
		var folds = 5
		var patterns, _, _ = neural.LoadPatternsFromCSVFile(filePath)
		var neuron = neural.NeuronUnit{Weights: make([]float64, len(patterns[0].Features)), Bias: bias, LearningRate: learningRate}
		var scores = validation.KFoldValidation(&neuron, patterns, epochs, folds, shuffle)
		var neuron2 = neural.NeuronUnit{Weights: make([]float64, len(patterns[0].Features)), Bias: bias, LearningRate: learningRate}
		var scores2 = validation.RandomSubsamplingValidation(&neuron2, patterns, percentage, epochs, folds, shuffle)

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"scores": scores,
		}).Info("Scores reached: ", scores)
		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"scores": scores2,
		}).Info("Scores reached: ", scores2)
	}
	if false {
		log.WithFields(log.Fields{
			"level": "info",
			"place": "main",
			"msg":   "multi layer perceptron train and test over iris dataset",
		}).Info("Compute backpropagation multi layer perceptron on sonar data set (binary classification problem)")

		var filePath = "./resources/iris.all_data.csv"
		var learningRate = 0.01
		var percentage = 0.67
		var shuffle = 1
		var epochs = 500
		var folds = 3
		var patterns, _, mapped = neural.LoadPatternsFromCSVFile(filePath)

		//input  layer : 4 neuron, represents the feature of Iris, more in general dimensions of pattern
		//hidden layer : 3 neuron, activation using sigmoid, number of neuron in hidden level
		// 2° hidden l : * neuron, insert number of level you want
		//output layer : 3 neuron, represents the class of Iris, more in general dimensions of mapped values
		var layers = []int{len(patterns[0].Features), 20, len(mapped)}

		var mlp = neural.PrepareMLPNet(layers, learningRate, neural.SigmoidTransfer, neural.SigmoidTransferDerivative)
		var scores = validation.MLPKFoldValidation(&mlp, patterns, epochs, folds, shuffle, mapped)
		var mlp2 = neural.PrepareMLPNet(layers, learningRate, neural.SigmoidTransfer, neural.SigmoidTransferDerivative)
		var scores2 = validation.MLPRandomSubsamplingValidation(&mlp2, patterns, percentage, epochs, folds, shuffle, mapped)

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"scores": scores,
		}).Info("Scores reached: ", scores)

		log.WithFields(log.Fields{
			"level":  "info",
			"place":  "main",
			"scores": scores2,
		}).Info("Scores reached: ", scores2)

	}
	if true {

		log.WithFields(log.Fields{
			"level": "info",
			"place": "main",
			"msg":   "multi layer perceptron train and test over iris dataset",
		}).Info("Compute training algorithm on elman network using iris data set (binary classification problem)")

		var learningRate = 0.01
		var epochs = 500
		var patterns = neural.CreateRandomPatternArray(8, 30)

		//input  layer : 4 neuron, represents the feature of Iris, more in general dimensions of pattern
		//hidden layer : 3 neuron, activation using sigmoid, number of neuron in hidden level
		// 2° hidden l : * neuron, insert number of level you want
		//output layer : 3 neuron, represents the class of Iris, more in general dimensions of mapped value
		//Multilayer perceptron model, with one hidden layer.
		var mlp = neural.PrepareElmanNet(len(patterns[0].Features)+10,
			10, len(patterns[0].MultipleExpectation), learningRate,
			neural.SigmoidTransfer, neural.SigmoidTransferDerivative)

		var mean, _ = validation.RNNValidation(&mlp, patterns, epochs)
		log.WithFields(log.Fields{
			"level":     "info",
			"place":     "main",
			"precision": util.Round(mean, .5, 2),
		}).Info("Scores reached: ", util.Round(mean, .5, 2))
	}
}
