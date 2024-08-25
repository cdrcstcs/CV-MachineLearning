package featureSelection

import(
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"strconv"
	"sort"
)

// FeatureSelectionResult represents the result of feature selection
type FeatureSelectionResult struct {
	FeatureIndices []int
	Scores         []float64
}

// loadData loads data from a CSV file
func loadData(filename string) ([][]float64, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	lines, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	numRows := len(lines)
	numCols := len(lines[0])

	X := make([][]float64, numRows-1)
	y := make([]float64, numRows-1)

	for i, line := range lines[1:] {
		X[i] = make([]float64, numCols-1)
		for j, val := range line[:numCols-1] {
			X[i][j], _ = strconv.ParseFloat(val, 64)
		}
		y[i], _ = strconv.ParseFloat(lines[i+1][numCols-1], 64)
	}

	return X, y, nil
}

// univariateFeatureSelection performs feature selection using univariate analysis
func univariateFeatureSelection(X [][]float64, y []float64, numFeatures int) FeatureSelectionResult {
	numSamples := len(X)
	numFeaturesAll := len(X[0])
	scores := make([]float64, numFeaturesAll)

	for i := 0; i < numFeaturesAll; i++ {
		featureValues := make([]float64, numSamples)
		for j := 0; j < numSamples; j++ {
			featureValues[j] = X[j][i]
		}
		scores[i] = calculateScore(featureValues, y)
	}

	// Rank features based on scores
	rankedIndices := make([]int, numFeaturesAll)
	for i := range rankedIndices {
		rankedIndices[i] = i
	}
	sortIndicesByScores(rankedIndices, scores)

	// Select top k features
	selectedIndices := rankedIndices[:numFeatures]
	selectedScores := make([]float64, numFeatures)
	for i, index := range selectedIndices {
		selectedScores[i] = scores[index]
	}

	return FeatureSelectionResult{FeatureIndices: selectedIndices, Scores: selectedScores}
}

// calculateScore calculates the score for a feature
func calculateScore(featureValues []float64, target []float64) float64 {
	var score float64
	// Implement a scoring method, e.g., correlation coefficient, mutual information, etc.
	// For simplicity, let's use the absolute correlation coefficient here
	correlation := math.Abs(correlationCoefficient(featureValues, target))
	score = correlation
	return score
}

// correlationCoefficient calculates the Pearson correlation coefficient between two variables
func correlationCoefficient(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return math.NaN()
	}

	n := len(x)
	meanX := mean(x)
	meanY := mean(y)
	var numerator, denomX, denomY float64

	for i := 0; i < n; i++ {
		numerator += (x[i] - meanX) * (y[i] - meanY)
		denomX += math.Pow(x[i]-meanX, 2)
		denomY += math.Pow(y[i]-meanY, 2)
	}

	denominator := math.Sqrt(denomX * denomY)
	if denominator == 0 {
		return 0 // If denominator is zero, correlation is zero
	}

	return numerator / denominator
}

// mean calculates the mean of a slice of values
func mean(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// sortIndicesByScores sorts feature indices based on their scores
func sortIndicesByScores(indices []int, scores []float64) {
	sort.Slice(indices, func(i, j int) bool {
		return scores[indices[i]] > scores[indices[j]]
	})
}

func main() {
	// Load data
	X, y, err := loadData("data.csv")
	if err != nil {
		fmt.Println("Error loading data:", err)
		return
	}

	// Perform univariate feature selection
	numFeaturesToSelect := 5 // Select top 5 features
	result := univariateFeatureSelection(X, y, numFeaturesToSelect)

	// Print selected feature indices and their scores
	fmt.Println("Selected Feature Indices:", result.FeatureIndices)
	fmt.Println("Corresponding Scores:", result.Scores)
}
