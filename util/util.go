package util

import (
	log "github.com/sirupsen/logrus"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

func init() {
	log.SetOutput(os.Stdout)
	log.SetLevel(log.DebugLevel)
}

// Random return pseudo random number in [min, max]
func Random(min, max int) int {
	max = max + 1
	rand.Seed(time.Now().Unix())
	return rand.Intn(max-min) + min
}

// StringInSlice looks for a string in slice.
// It returns true or false and position of string in slice (false, -1 if not found).
func StringInSlice(element string, slice []string) (bool, int) {
	for index, value := range slice {
		if value == element {
			return true, index
		}
	}
	return false, -1
}

// StringToFloat cast a slice of string element to a slice of float64 element.
// If passed mode is 0, for each error encountered in casting, passed default will be inserted
// in slice.
// If passed mode is 1, output slice will contain only correctly cast element.
// It returns slice converted.
func StringToFloat(slice []string, mode int, def float64) []float64 {
	var result []float64
	if mode == 0 {
		result = make([]float64, len(slice))
	}

	for index, value := range slice {
		casted, err := strconv.ParseFloat(value, 64)
		if mode == 0 {
			if err == nil {
				result[index] = casted
			} else {
				result[index] = def
			}
		} else {
			if err == nil {
				result = append(result, casted)
			}
		}
	}
	return result
}

// ScalarProduct compute scalar product between two float64 based slices.
// It returns a float64 value.
func ScalarProduct(a []float64, b []float64) float64 {
	if len(a) != len(b) {
		log.WithFields(log.Fields{
			"level":  "error",
			"place":  "mixed",
			"method": "ScalarProduct",
			"msg":    "scalar product between slices",
			"aLen":   len(a),
			"bLen":   len(b),
		}).Error("Failed to compute scalar product between slices: different length.")
		return -1.0
	}
	var result = 0.0
	for index, value := range a {
		result = result + (value * b[index])
	}
	return result
}

// MaxInSlice return max value in float64 slice
// It returns the max float64 value and index of max in slice.
func MaxInSlice(v []float64) (float64, int) {
	mv := 0.0
	mi := 0
	for i, e := range v {
		if e > mv {
			mv = e
			mi = i
		}
	}
	return mv, mi
}

func GenerateRandomIntWithBinaryDim(d int) int64 {
	rand.Seed(time.Now().UTC().UnixNano())
	return rand.Int63n(int64(2 ^ d))
}

func GenerateRandomBinaryInt(d int) []float64 {
	rand.Seed(time.Now().UTC().UnixNano())
	bn := rand.Int63n(int64(2 ^ d))
	bi := make([]float64, d)
	bs := strconv.FormatInt(bn, 2)
	zn := d - len(bs)
	for zIndex := 0; zIndex < d; zIndex++ {
		if zIndex < zn {
			bi[zIndex] = 0.0
		} else {
			bi[zIndex], _ = strconv.ParseFloat(string(bs[zIndex-zn]), 64)
		}
	}
	return bi
}

func ConvertIntToBinary(n int64, d int) []float64 {
	bs := strconv.FormatInt(n, 2)
	bi := make([]float64, d)
	zn := d - len(bs)
	if zn < 0 {
		log.Warning("Too small base")
		bi = make([]float64, len(bs))
		zn = 0
	}
	for zIndex := 0; zIndex < d; zIndex++ {
		if zIndex < zn {
			bi[zIndex] = 0.0
		} else {
			bi[zIndex], _ = strconv.ParseFloat(string(bs[zIndex-zn]), 64)
		}
	}
	return bi
}

func ConvertBinToInt(n []float64) int {
	bi := 0
	for i := len(n) - 1; i >= 0; i-- {
		bi = bi + (int(n[i]) * int(math.Pow(float64(2), float64(len(n)-i-1))))
	}
	return bi
}

func Round(val float64, roundOn float64, places int) (newVal float64) {
	var round float64
	pow := math.Pow(10, float64(places))
	digit := pow * val
	_, div := math.Modf(digit)
	if div >= roundOn {
		round = math.Ceil(digit)
	} else {
		round = math.Floor(digit)
	}
	newVal = round / pow
	return
}
