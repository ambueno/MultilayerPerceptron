package validation

import (
	"MultilayerPerceptron/neural"
	"MultilayerPerceptron/util"
	log "github.com/sirupsen/logrus"
	"math/rand"
	"time"
)

// TrainTestPatternsSplit split an array of patterns in training and testing.
// if shuffle is 0 the function takes the first percentage items as train and the other as test
// otherwise the patterns array is shuffled before partitioning
func TrainTestPatternsSplit(patterns []neural.Pattern, percentage float64, shuffle int) (train []neural.Pattern, test []neural.Pattern) {
	var splitPivot = int(float64(len(patterns)) * percentage)
	train = make([]neural.Pattern, splitPivot)
	test = make([]neural.Pattern, len(patterns)-splitPivot)

	if shuffle == 1 {
		rand.Seed(time.Now().UTC().UnixNano())
		perm := rand.Perm(len(patterns))
		for i := 0; i < splitPivot; i++ {
			train[i] = patterns[perm[i]]
		}
		for i := 0; i < len(patterns)-splitPivot; i++ {
			test[i] = patterns[perm[i]]
		}
	} else {
		train = patterns[:splitPivot]
		test = patterns[splitPivot:]
	}

	log.WithFields(log.Fields{
		"level":     "info",
		"msg":       "splitting completed",
		"trainSet":  len(train),
		"testSet: ": len(test),
	}).Info("Complete splitting train/test set.")

	return train, test
}

// TrainTestPatternSplit split an array of patterns in training and testing.
// if shuffle is 0 the function takes the first percentage items as train and the other as test
// otherwise the patterns array is shuffled before partitioning
func TrainTestPatternSplit(patterns []neural.Pattern, percentage float64, shuffle int) (train []neural.Pattern, test []neural.Pattern) {
	var splitPivot = int(float64(len(patterns)) * percentage)
	train = make([]neural.Pattern, splitPivot)
	test = make([]neural.Pattern, len(patterns)-splitPivot)

	if shuffle == 1 {
		rand.Seed(time.Now().UTC().UnixNano())
		perm := rand.Perm(len(patterns))

		for i := 0; i < splitPivot; i++ {
			train[i] = patterns[perm[i]]
		}
		for i := 0; i < len(patterns)-splitPivot; i++ {
			test[i] = patterns[perm[i]]
		}

	} else {
		train = patterns[:splitPivot]
		test = patterns[splitPivot:]
	}

	log.WithFields(log.Fields{
		"level":     "info",
		"msg":       "splitting completed",
		"trainSet":  len(train),
		"testSet: ": len(test),
	}).Info("Complete splitting train/test set.")

	return train, test
}

// KFoldPatternsSplit split an array of patterns in k subsets.
// if shuffle is 0 the function partitions the items maintaining the order
// otherwise the patterns array is shuffled before partitioning
func KFoldPatternsSplit(patterns []neural.Pattern, k int, shuffle int) [][]neural.Pattern {
	var size = len(patterns) / k
	var freeElements = len(patterns) % k

	folds := make([][]neural.Pattern, k)

	var perm []int
	if shuffle == 1 {
		rand.Seed(time.Now().UTC().UnixNano())
		perm = rand.Perm(len(patterns))
	}

	currSize := 0
	foldStart := 0
	curr := 0
	for f := 0; f < k; f++ {
		curr = foldStart
		currSize = size
		if f < freeElements {
			currSize++
		}
		folds[f] = make([]neural.Pattern, currSize)
		for i := 0; i < currSize; i++ {
			if shuffle == 1 {
				folds[f][i] = patterns[perm[curr]]
			} else {
				folds[f][i] = patterns[curr]
			}
			curr++
		}
		foldStart = curr
	}

	log.WithFields(log.Fields{
		"level":              "info",
		"msg":                "splitting completed",
		"numberOfFolds":      k,
		"meanFoldSize: ":     size,
		"consideredElements": (size * k) + freeElements,
	}).Info("Complete folds splitting.")

	return folds
}

// RandomSubsamplingValidation perform evaluation on neuron algorithm.
// It returns scores reached for each fold iteration.
func RandomSubsamplingValidation(neuron *neural.NeuronUnit, patterns []neural.Pattern, percentage float64, epochs int, folds int, shuffle int) []float64 {
	var scores, actual, predicted []float64
	var train, test []neural.Pattern
	scores = make([]float64, folds)

	for t := 0; t < folds; t++ {
		train, test = TrainTestPatternsSplit(patterns, percentage, shuffle)
		neural.TrainNeuron(neuron, train, epochs, 1)
		for _, pattern := range test {
			actual = append(actual, pattern.SingleExpectation)
			predicted = append(predicted, neural.Predict(neuron, &pattern))
		}
		_, percentageCorrect := neural.Accuracy(actual, predicted)
		scores[t] = percentageCorrect

		log.WithFields(log.Fields{
			"level":             "info",
			"place":             "validation",
			"method":            "RandomSubsamplingValidation",
			"foldNumber":        t,
			"trainSetLen":       len(train),
			"testSetLen":        len(test),
			"percentageCorrect": percentageCorrect,
		}).Info("Evaluation completed for current fold.")
	}

	acc := 0.0
	for i := 0; i < len(scores); i++ {
		acc += scores[i]
	}
	mean := acc / float64(len(scores))

	log.WithFields(log.Fields{
		"level":       "info",
		"place":       "validation",
		"method":      "RandomSubsamplingValidation",
		"folds":       folds,
		"trainSetLen": len(train),
		"testSetLen":  len(test),
		"meanScore":   mean,
	}).Info("Evaluation completed for all folds.")

	return scores
}

// KFoldValidation perform evaluation on neuron algorithm.
// It returns scores reached for each fold iteration.
func KFoldValidation(neuron *neural.NeuronUnit, patterns []neural.Pattern, epochs int, k int, shuffle int) []float64 {
	var scores, actual, predicted []float64
	var train, test []neural.Pattern
	scores = make([]float64, k)
	folds := KFoldPatternsSplit(patterns, k, shuffle)
	for t := 0; t < k; t++ {
		train = nil
		for i := 0; i < k; i++ {
			if i != t {
				train = append(train, folds[i]...)
			}
		}
		test = folds[t]
		neural.TrainNeuron(neuron, train, epochs, 1)
		for _, pattern := range test {
			actual = append(actual, pattern.SingleExpectation)
			predicted = append(predicted, neural.Predict(neuron, &pattern))
		}
		_, percentageCorrect := neural.Accuracy(actual, predicted)
		scores[t] = percentageCorrect

		log.WithFields(log.Fields{
			"level":             "info",
			"place":             "validation",
			"method":            "KFoldValidation",
			"foldNumber":        t,
			"trainSetLen":       len(train),
			"testSetLen":        len(test),
			"percentageCorrect": percentageCorrect,
		}).Info("Evaluation completed for current fold.")
	}

	acc := 0.0
	for i := 0; i < len(scores); i++ {
		acc += scores[i]
	}

	mean := acc / float64(len(scores))

	log.WithFields(log.Fields{
		"level":       "info",
		"place":       "validation",
		"method":      "KFoldValidation",
		"folds":       k,
		"trainSetLen": len(train),
		"testSetLen":  len(test),
		"meanScore":   mean,
	}).Info("Evaluation completed for all folds.")

	return scores
}

// MLPRandomSubsamplingValidation returns scores reached for each fold iteration.
func MLPRandomSubsamplingValidation(mlp *neural.MultiLayerNetwork, patterns []neural.Pattern, percentage float64, epochs int, folds int, shuffle int, mapped []string) []float64 {
	var scores, actual, predicted []float64
	var train, test []neural.Pattern
	scores = make([]float64, folds)

	for t := 0; t < folds; t++ {
		train, test = TrainTestPatternsSplit(patterns, percentage, shuffle)
		neural.MLPTrain(mlp, patterns, mapped, epochs)

		for _, pattern := range test {
			actual = append(actual, pattern.SingleExpectation)
			oOut := neural.Execute(mlp, &pattern)
			_, indexMaxOut := util.MaxInSlice(oOut)
			predicted = append(predicted, float64(indexMaxOut))
		}

		_, percentageCorrect := neural.Accuracy(actual, predicted)
		scores[t] = percentageCorrect

		log.WithFields(log.Fields{
			"level":             "info",
			"place":             "validation",
			"method":            "MLPRandomSubsamplingValidation",
			"foldNumber":        t,
			"trainSetLen":       len(train),
			"testSetLen":        len(test),
			"percentageCorrect": percentageCorrect,
		}).Info("Evaluation completed for current fold.")
	}

	acc := 0.0
	for i := 0; i < len(scores); i++ {
		acc += scores[i]
	}

	mean := acc / float64(len(scores))

	log.WithFields(log.Fields{
		"level":       "info",
		"place":       "validation",
		"method":      "MLPRandomSubsamplingValidation",
		"folds":       folds,
		"trainSetLen": len(train),
		"testSetLen":  len(test),
		"meanScore":   mean,
	}).Info("Evaluation completed for all folds.")

	return scores
}

// MLPKFoldValidation RandomSubsamplingValidation perform evaluation on neuron algorithm.
// It returns scores reached for each fold iteration.
func MLPKFoldValidation(mlp *neural.MultiLayerNetwork, patterns []neural.Pattern, epochs int, k int, shuffle int, mapped []string) []float64 {
	var scores, actual, predicted []float64
	var train, test []neural.Pattern
	scores = make([]float64, k)
	folds := KFoldPatternsSplit(patterns, k, shuffle)

	for t := 0; t < k; t++ {
		train = nil
		for i := 0; i < k; i++ {
			if i != t {
				train = append(train, folds[i]...)
			}
		}
		test = folds[t]
		neural.MLPTrain(mlp, patterns, mapped, epochs)
		for _, pattern := range test {
			// get actual
			actual = append(actual, pattern.SingleExpectation)
			// get output from network
			oOut := neural.Execute(mlp, &pattern)
			// get index of max output
			_, indexMaxOut := util.MaxInSlice(oOut)
			// add to predicted values
			predicted = append(predicted, float64(indexMaxOut))
		}
		_, percentageCorrect := neural.Accuracy(actual, predicted)
		scores[t] = percentageCorrect

		log.WithFields(log.Fields{
			"level":             "info",
			"place":             "validation",
			"method":            "MLPKFoldValidation",
			"foldNumber":        t,
			"trainSetLen":       len(train),
			"testSetLen":        len(test),
			"percentageCorrect": percentageCorrect,
		}).Info("Evaluation completed for current fold.")
	}
	acc := 0.0
	for i := 0; i < len(scores); i++ {
		acc += scores[i]
	}
	mean := acc / float64(len(scores))

	log.WithFields(log.Fields{
		"level":       "info",
		"place":       "validation",
		"method":      "MLPKFoldValidation",
		"folds":       k,
		"trainSetLen": len(train),
		"testSetLen":  len(test),
		"meanScore":   mean,
	}).Info("Evaluation completed for all folds.")

	return scores

}

// RNNValidation perform evaluation on neuron algorithm.
func RNNValidation(mlp *neural.MultiLayerNetwork, patterns []neural.Pattern, epochs int) (float64, []float64) {
	var scores []float64
	scores = make([]float64, len(patterns))
	neural.ElmanTrain(mlp, patterns, epochs)
	pCor := 0.0

	for pI, pattern := range patterns {
		oOut := neural.Execute(mlp, &pattern, 1)
		for oOutI, oOutV := range oOut {
			oOut[oOutI] = util.Round(oOutV, .5, 0)
		}
		log.WithFields(log.Fields{
			"a_p_b": pattern.Features,
			"rea_c": pattern.MultipleExpectation,
			"pre_c": oOut,
		}).Debug()

		_, pCor = neural.Accuracy(pattern.MultipleExpectation, oOut)
		scores[pI] = pCor
	}

	acc := 0.0
	for i := 0; i < len(scores); i++ {
		acc += scores[i]
	}
	mean := acc / float64(len(scores))

	log.WithFields(log.Fields{
		"level":       "info",
		"place":       "validation",
		"method":      "RNNValidation",
		"trainSetLen": len(patterns),
		"testSetLen":  len(patterns),
		"meanScore":   mean,
	}).Info("Evaluation completed for all patterns.")

	return mean, scores
}
