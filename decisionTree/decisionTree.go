package decisionTree

import(
	"fmt"
	"math"
	"sort"
)

// TreeNode represents a node in the decision tree
type TreeNode struct {
	AttributeIndex int
	Threshold      float64
	Category       string
	Left           *TreeNode
	Right          *TreeNode
	Prediction     int
}

// DecisionTree represents the decision tree model
type DecisionTree struct {
	Root *TreeNode
}

// Fit builds the decision tree model
func (dt *DecisionTree) Fit(X [][]float64, y []int, categoricalCols []bool) {
	dt.Root = buildTree(X, y, categoricalCols)
}

// Predict returns the predictions for input data
func (dt *DecisionTree) Predict(X [][]float64) []int {
	var predictions []int
	for _, sample := range X {
		predictions = append(predictions, dt.predictSample(sample))
	}
	return predictions
}

// predictSample returns the prediction for a single sample
func (dt *DecisionTree) predictSample(sample []float64) int {
	currentNode := dt.Root
	for currentNode.Left != nil && currentNode.Right != nil {
		if currentNode.AttributeIndex != -1 { // Split on numerical attribute
			if sample[currentNode.AttributeIndex] < currentNode.Threshold {
				currentNode = currentNode.Left
			} else {
				currentNode = currentNode.Right
			}
		} else { // Split on categorical attribute
			if sample[int(currentNode.Threshold)] == 0 {
				currentNode = currentNode.Left
			} else {
				currentNode = currentNode.Right
			}
		}
	}
	return currentNode.Prediction
}

// buildTree recursively constructs the decision tree
func buildTree(X [][]float64, y []int, categoricalCols []bool) *TreeNode {
	if len(uniqueElements(y)) == 1 {
		return &TreeNode{Prediction: y[0]}
	}
	numAttributes := len(X[0])
	minEntropy := math.Inf(1)
	var bestAttributeIndex int
	var bestThreshold float64
	var bestLeftX, bestRightX [][]float64
	var bestLeftY, bestRightY []int

	for i := 0; i < numAttributes; i++ {
		if categoricalCols[i] {
			// Split on categorical attribute
			leftX, rightX, leftY, rightY := splitCategorical(X, y, i)
			leftEntropy := entropy(leftY)
			rightEntropy := entropy(rightY)
			entropyWeighted := (float64(len(leftY))/float64(len(y)))*leftEntropy +
				(float64(len(rightY))/float64(len(y)))*rightEntropy
			if entropyWeighted < minEntropy {
				minEntropy = entropyWeighted
				bestAttributeIndex = -1 // Mark as categorical attribute split
				bestThreshold = float64(i) // Use the attribute index as category indicator
				bestLeftX, bestRightX = leftX, rightX
				bestLeftY, bestRightY = leftY, rightY
			}
		} else {
			// Split on numerical attribute
			attributeValues := getAttributeValues(X, i)
			sort.Float64s(attributeValues)
			for j := 0; j < len(attributeValues)-1; j++ {
				threshold := 0.5 * (attributeValues[j] + attributeValues[j+1])
				leftX, rightX, leftY, rightY := splitNumerical(X, y, i, threshold)
				leftEntropy := entropy(leftY)
				rightEntropy := entropy(rightY)
				entropyWeighted := (float64(len(leftY))/float64(len(y)))*leftEntropy +
					(float64(len(rightY))/float64(len(y)))*rightEntropy
				if entropyWeighted < minEntropy {
					minEntropy = entropyWeighted
					bestAttributeIndex = i
					bestThreshold = threshold
					bestLeftX, bestRightX = leftX, rightX
					bestLeftY, bestRightY = leftY, rightY
				}
			}
		}
	}
	if minEntropy == math.Inf(1) {
		return &TreeNode{Prediction: majorityVote(y)}
	}
	leftChild := buildTree(bestLeftX, bestLeftY, categoricalCols)
	rightChild := buildTree(bestRightX, bestRightY, categoricalCols)
	return &TreeNode{
		AttributeIndex: bestAttributeIndex,
		Threshold:      bestThreshold,
		Left:           leftChild,
		Right:          rightChild,
	}
}

// splitNumerical performs split for numerical attribute
func splitNumerical(X [][]float64, y []int, attributeIndex int, threshold float64) ([][]float64, [][]float64, []int, []int) {
	var leftX, rightX [][]float64
	var leftY, rightY []int

	for i, val := range X {
		if val[attributeIndex] < threshold {
			leftX = append(leftX, val)
			leftY = append(leftY, y[i])
		} else {
			rightX = append(rightX, val)
			rightY = append(rightY, y[i])
		}
	}
	return leftX, rightX, leftY, rightY
}

// splitCategorical performs split for categorical attribute
func splitCategorical(X [][]float64, y []int, attributeIndex int) ([][]float64, [][]float64, []int, []int) {
	var leftX, rightX [][]float64
	var leftY, rightY []int

	for i, val := range X {
		if val[attributeIndex] == 0 {
			leftX = append(leftX, val)
			leftY = append(leftY, y[i])
		} else {
			rightX = append(rightX, val)
			rightY = append(rightY, y[i])
		}
	}
	return leftX, rightX, leftY, rightY
}

// getAttributeValues returns unique values for a given attribute
func getAttributeValues(X [][]float64, attributeIndex int) []float64 {
	var attributeValues []float64
	for _, sample := range X {
		if !contains(attributeValues, sample[attributeIndex]) {
			attributeValues = append(attributeValues, sample[attributeIndex])
		}
	}
	return attributeValues
}

// contains checks if a slice contains a value
func contains(slice []float64, val float64) bool {
	for _, item := range slice {
		if item == val {
			return true
		}
	}
	return false
}

// uniqueElements returns unique elements in a slice
func uniqueElements(slice []int) []int {
	keys := make(map[int]bool)
	var unique []int
	for _, entry := range slice {
		if _, value := keys[entry]; !value {
			keys[entry] = true
			unique = append(unique, entry)
		}
	}
	return unique
}

// entropy calculates the entropy of a given set
func entropy(y []int) float64 {
	entropy := 0.0
	totalSamples := len(y)
	uniqueClasses := uniqueElements(y)
	for _, class := range uniqueClasses {
		proportion := float64(count(y, class)) / float64(totalSamples)
		entropy -= proportion * math.Log2(proportion)
	}
	return entropy
}

// count counts occurrences of an element in a slice
func count(slice []int, val int) int {
	count := 0
	for _, item := range slice {
		if item == val {
			count++
		}
	}
	return count
}

// majorityVote returns the class with the majority vote
func majorityVote(y []int) int {
	classCounts := make(map[int]int)
	for _, class := range y {
		classCounts[class]++
	}
	maxCount := 0
	majorityClass := 0
	for class, count := range classCounts {
		if count > maxCount {
			maxCount = count
			majorityClass = class
		}
	}
	return majorityClass
}

// Pruning functions...

func main() {
	X := [][]float64{
		{6.3, 3.3, 6.0, 2.5},
		{5.8, 2.7, 5.1, 1.9},
		{7.1, 3.0, 5.9, 2.1},
		{6.3, 2.9, 5.6, 1.8},
	}
	y := []int{2, 1, 2, 1}

	// Indicate which columns are categorical
	categoricalCols := []bool{false, false, false, false}

	// Create and fit the decision tree
	dt := DecisionTree{}
	dt.Fit(X, y, categoricalCols)

	// Predict class for new samples
	newSamples := [][]float64{
		{6.0, 3.0, 4.8, 1.8},
		{5.8, 2.8, 4.6, 1.6},
	}
	predictions := dt.Predict(newSamples)
	fmt.Println("Predictions:", predictions)
}
