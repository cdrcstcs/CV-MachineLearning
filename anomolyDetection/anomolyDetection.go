package anomolyDetection

import(
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
)

// Point represents a data point in the dataset
type Point []float64

// IsolationTreeNode represents a node in the isolation tree
type IsolationTreeNode struct {
	SplitFeature int
	SplitValue   float64
	Left         *IsolationTreeNode
	Right        *IsolationTreeNode
	Size         int
}

// IsolationForest represents an ensemble of isolation trees
type IsolationForest struct {
	Trees       []*IsolationTreeNode
	NumTrees    int
	MaxTreeDepth int
}

// NewIsolationForest initializes a new IsolationForest
func NewIsolationForest(numTrees, maxTreeDepth int) *IsolationForest {
	return &IsolationForest{
		Trees:       make([]*IsolationTreeNode, numTrees),
		NumTrees:    numTrees,
		MaxTreeDepth: maxTreeDepth,
	}
}

// Train builds isolation trees in the forest
func (forest *IsolationForest) Train(data [][]float64) {
	for i := 0; i < forest.NumTrees; i++ {
		tree := buildIsolationTree(data, 0, forest.MaxTreeDepth)
		forest.Trees[i] = tree
	}
}

// buildIsolationTree recursively builds an isolation tree
func buildIsolationTree(data [][]float64, currentDepth, maxDepth int) *IsolationTreeNode {
	if len(data) <= 1 || currentDepth >= maxDepth {
		return &IsolationTreeNode{Size: len(data)}
	}

	numFeatures := len(data[0])
	splitFeature := rand.Intn(numFeatures)
	minValue, maxValue := findMinMax(data, splitFeature)
	splitValue := rand.Float64() * (maxValue - minValue) + minValue

	leftData := make([][]float64, 0)
	rightData := make([][]float64, 0)

	for _, point := range data {
		if point[splitFeature] < splitValue {
			leftData = append(leftData, point)
		} else {
			rightData = append(rightData, point)
		}
	}

	left := buildIsolationTree(leftData, currentDepth+1, maxDepth)
	right := buildIsolationTree(rightData, currentDepth+1, maxDepth)

	return &IsolationTreeNode{
		SplitFeature: splitFeature,
		SplitValue:   splitValue,
		Left:         left,
		Right:        right,
		Size:         len(data),
	}
}

// findMinMax finds the minimum and maximum values of a feature in the dataset
func findMinMax(data [][]float64, featureIndex int) (min, max float64) {
	min = math.Inf(1)
	max = math.Inf(-1)
	for _, point := range data {
		if point[featureIndex] < min {
			min = point[featureIndex]
		}
		if point[featureIndex] > max {
			max = point[featureIndex]
		}
	}
	return min, max
}

// AnomalyScore calculates the anomaly score for a data point
func (forest *IsolationForest) AnomalyScore(point []float64) float64 {
	if forest.NumTrees == 0 {
		return 0
	}

	avgPathLength := 0.0
	for _, tree := range forest.Trees {
		avgPathLength += float64(tree.Traverse(point, 0))
	}
	avgPathLength /= float64(forest.NumTrees)

	return math.Pow(2, -avgPathLength/(2*averagePathLength(forest.MaxTreeDepth)))
}

// Traverse traverses the isolation tree and returns the path length for a data point
func (node *IsolationTreeNode) Traverse(point []float64, currentDepth int) int {
	if node == nil {
		return currentDepth
	}

	if currentDepth >= node.Size {
		return currentDepth
	}

	if node.Left == nil && node.Right == nil {
		return currentDepth + 1
	}

	if point[node.SplitFeature] < node.SplitValue {
		return node.Left.Traverse(point, currentDepth+1)
	}
	return node.Right.Traverse(point, currentDepth+1)
}

// averagePathLength returns the average path length for data points
func averagePathLength(numDataPoints int) float64 {
	if numDataPoints > 2 {
		return 2 * (math.Log(float64(numDataPoints-1)) + 0.5772156649 - float64(numDataPoints-1)/float64(numDataPoints))
	}
	return 1
}

// LoadDataFromFile loads data from a CSV file
func LoadDataFromFile(filename string) ([][]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	data := make([][]float64, len(records))
	for i, record := range records {
		data[i] = make([]float64, len(record))
		for j, value := range record {
			num, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, err
			}
			data[i][j] = num
		}
	}

	return data, nil
}

func main() {
	// Load data from file
	data, err := LoadDataFromFile("data.csv")
	if err != nil {
		fmt.Println("Error loading data:", err)
		return
	}

	// Number of trees in the forest
	numTrees := 100

	// Maximum depth of each tree
	maxTreeDepth := 10

	// Create and train the Isolation Forest
	forest := NewIsolationForest(numTrees, maxTreeDepth)
	forest.Train(data)

	// Calculate anomaly scores for sample points
	samplePoints := [][]float64{
		{3, 3},
		{11, 11},
		{0, 0},
	}

	// Print anomaly scores
	for _, point := range samplePoints {
		anomalyScore := forest.AnomalyScore(point)
		fmt.Printf("Anomaly score for point %v: %f\n", point, anomalyScore)
	}
}
