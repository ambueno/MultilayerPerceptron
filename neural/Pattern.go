package neural

type Pattern struct {
	// input dimension
	Features []float64
	//  filled by parser with input classification (in terms of belonging class)
	SingleRawExpectation string
	// the class which the pattern belongs
	SingleExpectation float64
	// multiple class classification problems
	MultipleExpectation []float64
}
