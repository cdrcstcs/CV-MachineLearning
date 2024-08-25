package dataNormalization

import(
	"fmt"
	"math"
)

// MinMaxScaler performs Min-Max normalization on a given feature
type MinMaxScaler struct {
	Min float64
	Max float64
}

// Fit computes the minimum and maximum values of the feature
func (scaler *MinMaxScaler) Fit(data []float64) {
	scaler.Min = data[0]
	scaler.Max = data[0]
	for _, val := range data {
		if val < scaler.Min {
			scaler.Min = val
		}
		if val > scaler.Max {
			scaler.Max = val
		}
	}
}

// Transform performs Min-Max normalization on a given value
func (scaler *MinMaxScaler) Transform(val float64) float64 {
	return (val - scaler.Min) / (scaler.Max - scaler.Min)
}

// ZScoreScaler performs Z-score normalization on a given feature
type ZScoreScaler struct {
	Mean float64
	StdDev float64
}

// Fit computes the mean and standard deviation of the feature
func (scaler *ZScoreScaler) Fit(data []float64) {
	sum := 0.0
	for _, val := range data {
		sum += val
	}
	scaler.Mean = sum / float64(len(data))

	sumSquaredDiff := 0.0
	for _, val := range data {
		diff := val - scaler.Mean
		sumSquaredDiff += diff * diff
	}
	scaler.StdDev = math.Sqrt(sumSquaredDiff / float64(len(data)))
}

// Transform performs Z-score normalization on a given value
func (scaler *ZScoreScaler) Transform(val float64) float64 {
	return (val - scaler.Mean) / scaler.StdDev
}

func main() {
	// Example data for normalization
	data := []float64{10, 20, 30, 40, 50}

	// Initialize Min-Max scaler
	minMaxScaler := MinMaxScaler{}
	minMaxScaler.Fit(data)

	// Normalize data using Min-Max scaler
	var normalizedMinMax []float64
	for _, val := range data {
		normalizedVal := minMaxScaler.Transform(val)
		normalizedMinMax = append(normalizedMinMax, normalizedVal)
	}

	// Initialize Z-score scaler
	zScoreScaler := ZScoreScaler{}
	zScoreScaler.Fit(data)

	// Normalize data using Z-score scaler
	var normalizedZScore []float64
	for _, val := range data {
		normalizedVal := zScoreScaler.Transform(val)
		normalizedZScore = append(normalizedZScore, normalizedVal)
	}

	// Print normalized data
	fmt.Println("Original Data:", data)
	fmt.Println("Min-Max Normalized Data:", normalizedMinMax)
	fmt.Println("Z-score Normalized Data:", normalizedZScore)
}
