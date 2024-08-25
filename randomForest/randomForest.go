package randomForest

import(
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
)

// RandomForest represents a Random Forest model
type RandomForest struct {
	Trees       []*DecisionTree
	NumTrees    int
	MaxDepth    int
	MaxFeatures int
	Task        string
}

// DecisionTree represents a single decision tree in the Random Forest
type DecisionTree struct {
	Root       *Node
	MaxDepth   int
	MaxFeatures int
	Task       string
}

// Node represents a node in the decision tree
type Node struct {
	FeatureIndex int
	Threshold    float64
	Prediction   float64
	Left         *Node
	Right        *Node
}

// NewRandomForest creates a new Random Forest model
func NewRandomForest(numTrees, maxDepth, maxFeatures int, task string) *RandomForest {
	return &RandomForest{
		Trees:       make([]*DecisionTree, numTrees),
		NumTrees:    numTrees,
		MaxDepth:    maxDepth,
		MaxFeatures: maxFeatures,
		Task:        task,
	}
}

// NewDecisionTree creates a new Decision Tree
func NewDecisionTree(maxDepth, maxFeatures int, task string) *DecisionTree {
	return &DecisionTree{
		Root:       nil,
		MaxDepth:   maxDepth,
		MaxFeatures: maxFeatures,
		Task:       task,
	}
}
func (dt *DecisionTree) traverseTree(sample []float64, node *Node) float64 {
	if node.Left == nil && node.Right == nil {
		return node.Prediction
	}
	if sample[node.FeatureIndex] < node.Threshold {
		return dt.traverseTree(sample, node.Left)
	}
	return dt.traverseTree(sample, node.Right)
}
func (dt *DecisionTree) majorityVote(predictions []float64) float64 {
	counts := make(map[float64]int)

	for _, prediction := range predictions {
		counts[prediction]++
	}

	var majorityPrediction float64
	maxCount := 0
	for prediction, count := range counts {
		if count > maxCount {
			maxCount = count
			majorityPrediction = prediction
		}
	}

	return majorityPrediction
}

// getLeafPrediction returns the prediction value for a leaf node
func (dt *DecisionTree) getLeafPrediction(y []float64) float64 {
	// For classification tasks, return the most frequent class label
	if dt.Task == "classification" {
		return dt.majorityVote(y)
	}
	// For regression tasks, return the mean of the target values
	return dt.mean(y)
}
// TrainRandomForest trains the Random Forest model
func (rf *RandomForest) TrainRandomForest(X [][]float64, y []float64) {
	numSamples := len(X)

	for i := 0; i < rf.NumTrees; i++ {
		// Bootstrap sampling for training data
		XSample, ySample := rf.bootstrapSample(X, y, numSamples)

		// Create a new decision tree
		tree := NewDecisionTree(rf.MaxDepth, rf.MaxFeatures, rf.Task)

		// Train the decision tree
		tree.TrainDecisionTree(XSample, ySample)

		// Add the trained tree to the Random Forest
		rf.Trees[i] = tree
	}
}

// PredictRandomForest predicts the output for a given input sample using the Random Forest model
func (rf *RandomForest) PredictRandomForest(sample []float64) float64 {
	predictions := make([]float64, rf.NumTrees)

	for i, tree := range rf.Trees {
		predictions[i] = tree.PredictDecisionTree(sample)
	}

	if rf.Task == "classification" {
		return rf.majorityVote(predictions)
	} else if rf.Task == "regression" {
		return rf.mean(predictions)
	}

	return math.NaN()
}

// bootstrapSample performs bootstrap sampling on the dataset
func (rf *RandomForest) bootstrapSample(X [][]float64, y []float64, numSamples int) ([][]float64, []float64) {
	XSample := make([][]float64, numSamples)
	ySample := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		index := rand.Intn(numSamples)
		XSample[i] = X[index]
		ySample[i] = y[index]
	}

	return XSample, ySample
}

// majorityVote returns the majority vote from the predictions
func (rf *RandomForest) majorityVote(predictions []float64) float64 {
	counts := make(map[float64]int)

	for _, prediction := range predictions {
		counts[prediction]++
	}

	var majorityPrediction float64
	maxCount := 0
	for prediction, count := range counts {
		if count > maxCount {
			maxCount = count
			majorityPrediction = prediction
		}
	}

	return majorityPrediction
}

// mean returns the mean of the predictions
func (rf *RandomForest) mean(predictions []float64) float64 {
	sum := 0.0
	for _, prediction := range predictions {
		sum += prediction
	}
	return sum / float64(len(predictions))
}

// TrainDecisionTree trains the Decision Tree model
func (dt *DecisionTree) TrainDecisionTree(X [][]float64, y []float64) {
	dt.Root = dt.buildTree(X, y, dt.MaxDepth)
}

// PredictDecisionTree predicts the output for a given input sample using the Decision Tree model
func (dt *DecisionTree) PredictDecisionTree(sample []float64) float64 {
	return dt.traverseTree(sample, dt.Root)
}

// buildTree recursively builds the decision tree
func (dt *DecisionTree) buildTree(X [][]float64, y []float64, depth int) *Node {
	if len(y) == 0 {
		return nil
	}
	if depth == 0 || dt.isSameClass(y) || dt.isSameValue(X) {
		return &Node{Prediction: dt.getLeafPrediction(y)}
	}

	numFeatures := len(X[0])
	selectedFeatures := dt.selectFeatures(numFeatures)

	bestFeatureIndex, bestThreshold := dt.findBestSplit(X, y, selectedFeatures)

	leftX, leftY, rightX, rightY := dt.splitData(X, y, bestFeatureIndex, bestThreshold)

	leftNode := dt.buildTree(leftX, leftY, depth-1)
	rightNode := dt.buildTree(rightX, rightY, depth-1)

	return &Node{
		FeatureIndex: bestFeatureIndex,
		Threshold:    bestThreshold,
		Left:         leftNode,
		Right:        rightNode,
	}
}

// selectFeatures randomly selects a subset of features
func (dt *DecisionTree) selectFeatures(numFeatures int) []int {
	selectedFeatures := make([]int, dt.MaxFeatures)
	for i := range selectedFeatures {
		selectedFeatures[i] = rand.Intn(numFeatures)
	}
	return selectedFeatures
}

// findBestSplit finds the best feature and threshold to split the data
func (dt *DecisionTree) findBestSplit(X [][]float64, y []float64, selectedFeatures []int) (int, float64) {
	bestFeatureIndex := -1
	bestThreshold := math.Inf(1)
	bestScore := math.Inf(-1)

	for _, featureIndex := range selectedFeatures {
		threshold, score := dt.findBestSplitForFeature(X, y, featureIndex)
		if score > bestScore {
			bestFeatureIndex = featureIndex
			bestThreshold = threshold
			bestScore = score
		}
	}

	return bestFeatureIndex, bestThreshold
}

// findBestSplitForFeature finds the best threshold to split the data for a given feature
func (dt *DecisionTree) findBestSplitForFeature(X [][]float64, y []float64, featureIndex int) (float64, float64) {
	var bestThreshold float64
	bestScore := math.Inf(-1)

	// Sort feature values
	featureValues := make([]float64, len(X))
	for i := range X {
		featureValues[i] = X[i][featureIndex]
	}
	sort.Float64s(featureValues)

	// Calculate potential split points
	splitPoints := make([]float64, len(X)-1)
	for i := 0; i < len(X)-1; i++ {
		splitPoints[i] = (featureValues[i] + featureValues[i+1]) / 2
	}

	// Evaluate split points
	for _, threshold := range splitPoints {
		leftY := make([]float64, 0)
		rightY := make([]float64, 0)

		for i, value := range X {
			if value[featureIndex] < threshold {
				leftY = append(leftY, y[i])
			} else {
				rightY = append(rightY, y[i])
			}
		}

		score := dt.calculateScore(leftY, rightY)
		if score > bestScore {
			bestThreshold = threshold
			bestScore = score
		}
	}

	return bestThreshold, bestScore
}

// calculateScore calculates the score for a given split
func (dt *DecisionTree) calculateScore(leftY, rightY []float64) float64 {
	leftSize := float64(len(leftY))
	rightSize := float64(len(rightY))
	totalSize := leftSize + rightSize

	if dt.Task == "classification" {
		leftGini := dt.giniImpurity(leftY)
		rightGini := dt.giniImpurity(rightY)
		weightedGini := (leftSize/totalSize)*leftGini + (rightSize/totalSize)*rightGini
		return -weightedGini // Minimize Gini impurity
	} else if dt.Task == "regression" {
		leftMSE := dt.meanSquaredError(leftY)
		rightMSE := dt.meanSquaredError(rightY)
		weightedMSE := (leftSize/totalSize)*leftMSE + (rightSize/totalSize)*rightMSE
		return -weightedMSE // Minimize mean squared error
	}

	return math.NaN()
}

// giniImpurity calculates the Gini impurity for a given set of labels
func (dt *DecisionTree) giniImpurity(y []float64) float64 {
	classCounts := make(map[float64]int)
	for _, label := range y {
		classCounts[label]++
	}

	var impurity float64
	for _, count := range classCounts {
		prob := float64(count) / float64(len(y))
		impurity += prob * (1 - prob)
	}
	return impurity
}

// meanSquaredError calculates the mean squared error for a given set of values
func (dt *DecisionTree) meanSquaredError(y []float64) float64 {
	mean := dt.mean(y)
	var mse float64
	for _, value := range y {
		mse += math.Pow(value-mean, 2)
	}
	return mse / float64(len(y))
}

// mean calculates the mean of a slice of values
func (dt *DecisionTree) mean(y []float64) float64 {
	sum := 0.0
	for _, value := range y {
		sum += value
	}
	return sum / float64(len(y))
}

// isSameClass checks if all elements in y belong to the same class
func (dt *DecisionTree) isSameClass(y []float64) bool {
	for i := 1; i < len(y); i++ {
		if y[i] != y[0] {
			return false
		}
	}
	return true
}

// isSameValue checks if all elements in X belong to the same value
func (dt *DecisionTree) isSameValue(X [][]float64) bool {
	for i := 1; i < len(X); i++ {
		for j := 1; j < len(X[i]); j++ {
			if X[i][j] != X[0][j] {
				return false
			}
		}
	}
	return true
}

// splitData splits the dataset into left and right based on the threshold
func (dt *DecisionTree) splitData(X [][]float64, y []float64, featureIndex int, threshold float64) ([][]float64, []float64, [][]float64, []float64) {
	leftX := make([][]float64, 0)
	leftY := make([]float64, 0)
	rightX := make([][]float64, 0)
	rightY := make([]float64, 0)

	for i := range X {
		if X[i][featureIndex] < threshold {
			leftX = append(leftX, X[i])
			leftY = append(leftY, y[i])
		} else {
			rightX = append(rightX, X[i])
			rightY = append(rightY, y[i])
		}
	}

	return leftX, leftY, rightX, rightY
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
			if val == "?" {
				X[i][j] = math.NaN()
			} else {
				X[i][j], _ = strconv.ParseFloat(val, 64)
			}
		}
		y[i], _ = strconv.ParseFloat(strings.TrimSpace(line[numCols-1]), 64)
	}

	return X, y, nil
}

func main() {
	// Load data
	X, y, err := loadData("data.csv")
	if err != nil {
		fmt.Println("Error loading data:", err)
		return
	}

	// Replace missing values with mean imputation
	X = imputeMissingValues(X)

	// Split data into training and testing sets
	XTrain, yTrain, XTest, yTest := splitData(X, y, 0.8)

	// Create and train Random Forest
	rf := NewRandomForest(10, 5, 2, "classification")
	rf.TrainRandomForest(XTrain, yTrain)

	// Evaluate Random Forest
	accuracy := evaluateRandomForest(rf, XTest, yTest)
	fmt.Println("Accuracy:", accuracy)
}

// imputeMissingValues replaces missing values with mean imputation
func imputeMissingValues(X [][]float64) [][]float64 {
	numRows := len(X)
	numCols := len(X[0])
	means := make([]float64, numCols)

	// Calculate mean for each feature
	for j := 0; j < numCols; j++ {
		sum := 0.0
		count := 0
		for i := 0; i < numRows; i++ {
			if !math.IsNaN(X[i][j]) {
				sum += X[i][j]
				count++
			}
		}
		means[j] = sum / float64(count)
	}

	// Replace missing values with mean
	for i := 0; i < numRows; i++ {
		for j := 0; j < numCols; j++ {
			if math.IsNaN(X[i][j]) {
				X[i][j] = means[j]
			}
		}
	}

	return X
}

// splitData splits the data into training and testing sets
func splitData(X [][]float64, y []float64, splitRatio float64) ([][]float64, []float64, [][]float64, []float64) {
	numTrain := int(float64(len(X)) * splitRatio)
	XTrain := X[:numTrain]
	yTrain := y[:numTrain]
	XTest := X[numTrain:]
	yTest := y[numTrain:]
	return XTrain, yTrain, XTest, yTest
}

// evaluateRandomForest evaluates the performance of the Random Forest model
func evaluateRandomForest(rf *RandomForest, XTest [][]float64, yTest []float64) float64 {
	correct := 0

	for i, sample := range XTest {
		prediction := rf.PredictRandomForest(sample)
		if prediction == yTest[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(XTest))
}
