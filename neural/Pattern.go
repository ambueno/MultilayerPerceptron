package neural

import (
	"MultilayerPerceptron/util"
	"encoding/csv"
	log "github.com/sirupsen/logrus"
	"io"
	ioutil "io/ioutil"
	"os"
	"strings"
)

func init() {
	log.SetOutput(os.Stdout)
	log.SetLevel(log.InfoLevel)
}

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

// LoadPatternsFromCSVFile load a CSV dataset into an array of Pattern.
func LoadPatternsFromCSVFile(filePath string) ([]Pattern, error, []string) {
	var patterns []Pattern
	fileContent, errorValue := ioutil.ReadFile(filePath)

	if errorValue != nil {
		log.WithFields(log.Fields{
			"level":      "fatal",
			"place":      "patterns",
			"method":     "LoadPatternsFromCSVFile",
			"msg":        "reading file in specific path",
			"filePath":   filePath,
			"errorValue": errorValue,
		}).Fatal("Failed to read file in specified path.")
		return patterns, errorValue, nil
	}
	pointer := csv.NewReader(strings.NewReader(string(fileContent)))
	var lineCounter = 0
	for {
		line, errorValue := pointer.Read()
		log.WithFields(log.Fields{
			"level":  "debug",
			"place":  "patterns",
			"method": "LoadPatternsFromCSVFile",
			"line":   line,
		}).Debug()
		if errorValue == io.EOF {
			log.WithFields(log.Fields{
				"level":    "info",
				"place":    "patterns",
				"method":   "LoadPatternsFromCSVFile",
				"readData": len(patterns),
				"msg":      "File reading completed.",
			}).Info("File reading completed.")
			break
		}
		if errorValue != nil {
			log.WithFields(log.Fields{
				"level":       "errorValue",
				"place":       "patterns",
				"method":      "LoadPatternsFromCSVFile",
				"msg":         "parsing file in specific line number",
				"lineCounter": lineCounter,
				"errorValue":  errorValue,
			}).Error("Failed to parse line.")
			return patterns, errorValue, nil
		}
		var floatingValues = util.StringToFloat(line, 1, -1.0)
		patterns = append(
			patterns,
			Pattern{Features: floatingValues, SingleRawExpectation: line[len(line)-1]})
		lineCounter = lineCounter + 1
	}
	mapped := RawExpectedConversion(patterns)
	return patterns, nil, mapped
}

// RawExpectedConversion converts (string) raw expected values in patterns
// training / testing sets to float64 values
// It works on pattern struct (pointer) passed. It doesn't return anything
func RawExpectedConversion(patterns []Pattern) []string {
	var rawExpectedValues []string
	for _, pattern := range patterns {
		check, _ := util.StringInSlice(pattern.SingleRawExpectation, rawExpectedValues)
		if !check {
			rawExpectedValues = append(rawExpectedValues, pattern.SingleRawExpectation)
		}
		log.WithFields(log.Fields{
			"level":            "debug",
			"place":            "patterns",
			"msg":              "raw class extraction",
			"rawExpectedAdded": pattern.SingleRawExpectation,
		}).Debug()
	}

	log.WithFields(log.Fields{
		"level":             "info",
		"place":             "patterns",
		"msg":               "raw class extraction completed",
		"numberOfRawUnique": len(rawExpectedValues),
	}).Info("Complete SingleRawExpectation value set filling.")

	for index := range patterns {
		for mapped, value := range rawExpectedValues {
			if strings.Compare(value, patterns[index].SingleRawExpectation) == 0 {
				patterns[index].SingleExpectation = float64(mapped)
			}
		}
	}
	return rawExpectedValues
}

// CreateRandomPatternArray load a CSV dataset into an array of Pattern.
func CreateRandomPatternArray(d int, k int) []Pattern {
	var patterns []Pattern
	var i = 0
	for i < k {

		a := util.GenerateRandomIntWithBinaryDim(d)
		b := util.GenerateRandomIntWithBinaryDim(d)
		c := a + b
		log.WithFields(log.Fields{
			"ai": a,
			"as": util.ConvertIntToBinary(a, d),
			"bi": b,
			"bs": util.ConvertIntToBinary(b, d),
			"ci": c,
			"cs": util.ConvertIntToBinary(c, d+1),
		}).Debug()
		ab := util.ConvertIntToBinary(a, d)
		bb := util.ConvertIntToBinary(b, d)
		for _, v := range bb {
			ab = append(ab, v)
		}

		patterns = append(patterns, Pattern{Features: ab, MultipleExpectation: util.ConvertIntToBinary(c, d+1)})
		i = i + 1

	}

	return patterns
}
