package oneR

import(
	"fmt"
	"math"
)

// DataPoint represents a data instance with features and a target label
type DataPoint struct {
	Features []float64
	Target   string
}

// OneRModel represents the One-R model with a single rule
type OneRModel struct {
	Rule       string
	FeatureIdx int
}

// TrainOneR trains a One-R model on the provided dataset
func TrainOneR(data []DataPoint) OneRModel {
	bestError := math.Inf(1)
	var bestRule string
	var bestFeatureIdx int

	// Iterate over each feature
	for featureIdx := range data[0].Features {
		// Calculate mode for each unique value of the feature
		counts := make(map[float64]map[string]int)
		for _, point := range data {
			featureValue := point.Features[featureIdx]
			if counts[featureValue] == nil {
				counts[featureValue] = make(map[string]int)
			}
			counts[featureValue][point.Target]++
		}

		// Find the most frequent class for each unique feature value
		var totalErrors int
		var rule string
		for value, classCounts := range counts {
			mostFrequentClass := ""
			maxCount := 0
			for class, count := range classCounts {
				if count > maxCount {
					maxCount = count
					mostFrequentClass = class
				}
			}
			for _, point := range data {
				if point.Features[featureIdx] == value && point.Target != mostFrequentClass {
					totalErrors++
				}
			}
			if rule == "" {
				rule = fmt.Sprintf("If Feature[%d] == %.2f, predict %s", featureIdx, value, mostFrequentClass)
			}
		}

		// Update the best rule if the current one has fewer errors
		if float64(totalErrors) < bestError {
			bestError = float64(totalErrors)
			bestRule = rule
			bestFeatureIdx = featureIdx
		}
	}

	return OneRModel{
		Rule:       bestRule,
		FeatureIdx: bestFeatureIdx,
	}
}

// PredictOneR predicts the target label for a given data instance using the One-R model
func PredictOneR(model OneRModel, point DataPoint) string {
	if point.Features[model.FeatureIdx] == 1 {
		return "1"
	}
	return "0"
}

func main() {
	// Example dataset
	data := []DataPoint{
		{Features: []float64{0}, Target: "0"},
		{Features: []float64{1}, Target: "1"},
		{Features: []float64{1}, Target: "1"},
		{Features: []float64{0}, Target: "1"},
		{Features: []float64{1}, Target: "0"},
		{Features: []float64{0}, Target: "1"},
	}

	// Train One-R model
	model := TrainOneR(data)
	fmt.Println("One-R Rule:", model.Rule)

	// Example prediction
	testInstance := DataPoint{Features: []float64{0}, Target: ""}
	prediction := PredictOneR(model, testInstance)
	fmt.Println("Prediction for", testInstance.Features, ":", prediction)
}
