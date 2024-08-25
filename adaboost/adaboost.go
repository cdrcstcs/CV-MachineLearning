package adaboost

import(
	"fmt"
	"math"
)

type AdaBoost struct {
	WeakLearners []WeakLearner
	Alpha        []float64
}

type WeakLearner struct {
	FeatureIndex int
	Threshold    float64
	Direction    int
}

func NewAdaBoost() *AdaBoost {
	return &AdaBoost{}
}

func (adaboost *AdaBoost) Train(X [][]float64, y []float64, numIterations int) {
	numSamples := len(X)
	numFeatures := len(X[0])
	weights := make([]float64, numSamples)

	// Initialize weights
	for i := range weights {
		weights[i] = 1.0 / float64(numSamples)
	}

	for t := 0; t < numIterations; t++ {
		weakLearner := WeakLearner{}
		errorRate := math.MaxFloat64

		// Find the best weak learner
		for j := 0; j < numFeatures; j++ {
			for _, direction := range []int{-1, 1} {
				for _, threshold := range findThresholds(X, j) {
					prediction := makePrediction(X, j, threshold, direction)
					weightedError := calculateWeightedError(weights, y, prediction)

					if weightedError < errorRate {
						errorRate = weightedError
						weakLearner = WeakLearner{j, threshold, direction}
					}
				}
			}
		}

		// Update alpha
		alpha := 0.5 * math.Log((1-errorRate)/errorRate)
		adaboost.Alpha = append(adaboost.Alpha, alpha)

		// Update weights
		z := 0.0
		for i := range weights {
			prediction := makePrediction(X, weakLearner.FeatureIndex, weakLearner.Threshold, weakLearner.Direction)
			isCorrect := 1.0
			if prediction[i] != y[i] {
				isCorrect = -1.0
			}
			weights[i] *= math.Exp(isCorrect * alpha * y[i] * prediction[i])
			z += weights[i]
		}

		// Normalize weights
		for i := range weights {
			weights[i] /= z
		}

		adaboost.WeakLearners = append(adaboost.WeakLearners, weakLearner)
	}
}

func findThresholds(X [][]float64, featureIndex int) []float64 {
	thresholds := make(map[float64]bool)
	for _, sample := range X {
		thresholds[sample[featureIndex]] = true
	}
	var result []float64
	for key := range thresholds {
		result = append(result, key)
	}
	return result
}

func makePrediction(X [][]float64, featureIndex int, threshold float64, direction int) []float64 {
	var predictions []float64
	for _, sample := range X {
		if sample[featureIndex]*float64(direction) < threshold*float64(direction) {
			predictions = append(predictions, -1.0)
		} else {
			predictions = append(predictions, 1.0)
		}
	}
	return predictions
}

func calculateWeightedError(weights []float64, y []float64, predictions []float64) float64 {
	totalWeight := 0.0
	weightedError := 0.0
	for i, prediction := range predictions {
		totalWeight += weights[i]
		if prediction != y[i] {
			weightedError += weights[i]
		}
	}
	return weightedError / totalWeight
}

func (adaboost *AdaBoost) Predict(X [][]float64) []float64 {
	numSamples := len(X)
	numIterations := len(adaboost.WeakLearners)
	predictions := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		prediction := 0.0
		for t := 0; t < numIterations; t++ {
			weakLearner := adaboost.WeakLearners[t]
			alpha := adaboost.Alpha[t]
			if X[i][weakLearner.FeatureIndex]*float64(weakLearner.Direction) < weakLearner.Threshold*float64(weakLearner.Direction) {
				prediction += -1.0 * alpha
			} else {
				prediction += 1.0 * alpha
			}
		}
		if prediction < 0 {
			predictions[i] = -1.0
		} else {
			predictions[i] = 1.0
		}
	}
	return predictions
}

func main() {
	X := [][]float64{
		{1, 2},
		{2, 3},
		{3, 4},
		{4, 5},
	}
	y := []float64{-1, -1, 1, 1}

	adaboost := NewAdaBoost()
	adaboost.Train(X, y, 10)

	fmt.Println("Predictions:", adaboost.Predict(X))
}
