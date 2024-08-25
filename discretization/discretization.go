package discretization

import (
	"fmt"
	"sort"
)

// EqualWidthDiscretization divides continuous values into bins of equal width
func EqualWidthDiscretization(data []float64, numBins int) []string {
	sort.Float64s(data)

	minVal := data[0]
	maxVal := data[len(data)-1]
	binWidth := (maxVal - minVal) / float64(numBins)

	bins := make([]string, numBins)
	for i := 0; i < numBins; i++ {
		binStart := minVal + float64(i)*binWidth
		binEnd := binStart + binWidth
		bins[i] = fmt.Sprintf("%.2f - %.2f", binStart, binEnd)
	}

	discretizedData := make([]string, len(data))
	for i, val := range data {
		binIndex := int((val - minVal) / binWidth)
		if binIndex == numBins {
			binIndex--
		}
		discretizedData[i] = bins[binIndex]
	}

	return discretizedData
}

// EqualFrequencyDiscretization divides continuous values into bins of equal frequency
func EqualFrequencyDiscretization(data []float64, numBins int) []string {
	sort.Float64s(data)

	binSize := len(data) / numBins
	bins := make([]string, numBins)

	for i := 0; i < numBins-1; i++ {
		binStart := i * binSize
		binEnd := (i + 1) * binSize
		bins[i] = fmt.Sprintf("%d - %d", int(data[binStart]), int(data[binEnd-1]))
	}

	// Handle the last bin separately if the number of data points is not divisible evenly by numBins
	lastBinStart := (numBins - 1) * binSize
	bins[numBins-1] = fmt.Sprintf("%d - %d", int(data[lastBinStart]), int(data[len(data)-1]))

	discretizedData := make([]string, len(data))
	for i := range data {
		binIndex := i / binSize
		discretizedData[i] = bins[binIndex]
	}

	return discretizedData
}

func main() {
	// Example data for discretization
	data := []float64{22, 35, 45, 28, 60, 18, 40, 55, 38, 29}
	numBins := 3

	// Equal width discretization (binning)
	binnedWidth := EqualWidthDiscretization(data, numBins)
	fmt.Println("Equal Width Discretization (Binning):", binnedWidth)

	// Equal frequency discretization
	binnedFreq := EqualFrequencyDiscretization(data, numBins)
	fmt.Println("Equal Frequency Discretization:", binnedFreq)
}
