package gradientBoost

import(
	"fmt"
	"math"
)

type GradientBoosting struct {
	Trees         []*RegressionTree
	LearningRate float64
}

type RegressionTree struct {
	Root *Node
}

type Node struct {
	FeatureIndex int
	Threshold    float64
	Value        float64
	Left         *Node
	Right        *Node
}

func NewGradientBoosting(learningRate float64) *GradientBoosting {
	return &GradientBoosting{
		LearningRate: learningRate,
	}
}

func (gb *GradientBoosting) Train(X [][]float64, y []float64, numIterations int) {
	numSamples := len(X)
	predictions := make([]float64, numSamples)

	// Initialize predictions with the mean of y
	mean := calculateMean(y)
	for i := range predictions {
		predictions[i] = mean
	}

	for t := 0; t < numIterations; t++ {
		// Calculate residuals
		residuals := calculateResiduals(y, predictions)

		// Train a regression tree on the residuals
		tree := gb.trainRegressionTree(X, residuals)

		// Update predictions
		for i, sample := range X {
			predictions[i] += gb.LearningRate * tree.Predict(sample)
		}

		// Add the trained tree to the ensemble
		gb.Trees = append(gb.Trees, tree)
	}
}
func (tree *RegressionTree) Predict(sample []float64) float64 {
	return tree.Root.traverseTree(sample)
}

func calculateMean(values []float64) float64 {
	sum := 0.0
	for _, value := range values {
		sum += value
	}
	return sum / float64(len(values))
}

func calculateResiduals(y, predictions []float64) []float64 {
	residuals := make([]float64, len(y))
	for i := range y {
		residuals[i] = y[i] - predictions[i]
	}
	return residuals
}

func (gb *GradientBoosting) trainRegressionTree(X [][]float64, y []float64) *RegressionTree {
	tree := &RegressionTree{}
	tree.Root = gb.buildTree(X, y, 0)
	return tree
}

func (gb *GradientBoosting) buildTree(X [][]float64, y []float64, depth int) *Node {
	if depth >= 2 {
		return &Node{Value: calculateMean(y)}
	}

	bestFeatureIndex := 0
	bestThreshold := 0.0
	bestScore := math.Inf(1)

	numSamples := len(X)
	numFeatures := len(X[0])

	for i := 0; i < numFeatures; i++ {
		for j := 0; j < numSamples; j++ {
			_, leftY, _, rightY := splitData(X, y, i, X[j][i])
			score := calculateScore(leftY, rightY)
			if score < bestScore {
				bestFeatureIndex = i
				bestThreshold = X[j][i]
				bestScore = score
			}
		}
	}

	leftX, leftY, rightX, rightY := splitData(X, y, bestFeatureIndex, bestThreshold)
	leftNode := gb.buildTree(leftX, leftY, depth+1)
	rightNode := gb.buildTree(rightX, rightY, depth+1)

	return &Node{
		FeatureIndex: bestFeatureIndex,
		Threshold:    bestThreshold,
		Left:         leftNode,
		Right:        rightNode,
	}
}

func calculateScore(leftY, rightY []float64) float64 {
	meanLeft := calculateMean(leftY)
	meanRight := calculateMean(rightY)

	var score float64
	for _, value := range leftY {
		score += math.Pow(value-meanLeft, 2)
	}
	for _, value := range rightY {
		score += math.Pow(value-meanRight, 2)
	}
	return score
}

func (gb *GradientBoosting) Predict(sample []float64) float64 {
	prediction := 0.0
	for _, tree := range gb.Trees {
		prediction += gb.LearningRate * tree.Root.traverseTree(sample)
	}
	return prediction
}

func (node *Node) traverseTree(sample []float64) float64 {
	if node.Left == nil && node.Right == nil {
		return node.Value
	}
	if sample[node.FeatureIndex] < node.Threshold {
		return node.Left.traverseTree(sample)
	}
	return node.Right.traverseTree(sample)
}

func splitData(X [][]float64, y []float64, featureIndex int, threshold float64) (leftX [][]float64, leftY []float64, rightX [][]float64, rightY []float64) {
	for i := range X {
		if X[i][featureIndex] < threshold {
			leftX = append(leftX, X[i])
			leftY = append(leftY, y[i])
		} else {
			rightX = append(rightX, X[i])
			rightY = append(rightY, y[i])
		}
	}
	return
}

func main() {
	X := [][]float64{
		{1, 2},
		{2, 3},
		{3, 4},
		{4, 5},
	}
	y := []float64{1, 2, 3, 4}

	gb := NewGradientBoosting(0.1)
	gb.Train(X, y, 100)

	fmt.Println("Predictions:")
	for _, sample := range X {
		fmt.Println(gb.Predict(sample))
	}
}
