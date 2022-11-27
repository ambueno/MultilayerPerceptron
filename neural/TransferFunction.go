package neural

import (
	"math"
)

type transferFunction func(float64) float64

func HeavisideTransfer(d float64) float64 {
	if d >= 0.0 {
		return 1.0
	}
	return 0.0
}

func HeavisideTransferDerivative(d float64) float64 {
	return 1.0
}

func SigmoidTransfer(d float64) float64 {
	return 1 / (1 + math.Pow(math.E, -d))
}

func SigmoidTransferDerivative(d float64) float64 {
	return 1.0
}

func HyperbolicTransfer(d float64) float64 {
	return math.Tanh(d)
}

func HyperbolicTransferDerivative(d float64) float64 {
	return 1 - math.Pow(d, 2)
}
