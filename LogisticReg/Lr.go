package LogisticReg

import(
	"fmt"
	"math"
)

// LogisticRegression struct represents the logistic regression model
type LogisticRegression struct {
	Weights []float64 // Coefficients for the logistic regression model
	LearningRate float64 // Learning rate for gradient descent
	Epochs int // Number of training epochs
}

// NewLogisticRegression initializes a new logistic regression model with default parameters
func NewLogisticRegression() *LogisticRegression {
	return &LogisticRegression{
		LearningRate: 0.01,
		Epochs:       1000,
	}
}

// Sigmoid function computes the sigmoid of a value
func Sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// Predict computes the predicted probability for a given input
func (lr *LogisticRegression) Predict(X []float64) float64 {
	var y float64
	for i := range X {
		y += lr.Weights[i] * X[i]
	}
	return Sigmoid(y)
}

// Train fits the logistic regression model to the training data
func (lr *LogisticRegression) Train(X [][]float64, y []int) {
	// Initialize weights
	lr.Weights = make([]float64, len(X[0]))
	for i := range lr.Weights {
		lr.Weights[i] = 0.0
	}

	// Gradient Descent
	for epoch := 0; epoch < lr.Epochs; epoch++ {
		for i, xi := range X {
			predicted := lr.Predict(xi)
			error := float64(y[i]) - predicted
			for j := range lr.Weights {
				lr.Weights[j] += lr.LearningRate * error * xi[j]
			}
		}
	}
}

func main() {
	// Example usage
	X := [][]float64{{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}}
	y := []int{0, 0, 1, 1, 1}

	// Initialize logistic regression model
	lr := NewLogisticRegression()

	// Train the model
	lr.Train(X, y)

	// Print trained weights
	fmt.Println("Trained Weights:", lr.Weights)

	// Predict new samples
	newSample := []float64{2, 3}
	prediction := lr.Predict(newSample)
	fmt.Println("Prediction for", newSample, ":", prediction)
}
