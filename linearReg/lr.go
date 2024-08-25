package linearReg

import(
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
)

// LinearRegression performs linear regression to find the best-fit line.
type LinearRegression struct {
	theta    []float64 // Parameters (theta0, theta1, ..., thetaN)
	features int       // Number of input features
}

// Fit trains the linear regression model using the provided input and output data.
func (lr *LinearRegression) Fit(X [][]float64, y []float64, alpha float64, numIterations int) {
	m := len(X)     // Number of training examples
	lr.features = len(X[0])

	// Initialize theta values
	lr.theta = make([]float64, lr.features+1)

	// Perform gradient descent
	for iteration := 0; iteration < numIterations; iteration++ {
		gradients := make([]float64, lr.features+1)

		// Compute gradients
		for i := 0; i < m; i++ {
			yPred := lr.Predict(X[i])
			error := yPred - y[i]
			gradients[0] += error

			for j := 1; j <= lr.features; j++ {
				gradients[j] += error * X[i][j-1]
			}
		}

		// Update theta values
		for j := 0; j <= lr.features; j++ {
			gradients[j] /= float64(m)
			lr.theta[j] -= alpha * gradients[j]
		}
	}
}

// Predict predicts the output for a given input vector.
func (lr *LinearRegression) Predict(x []float64) float64 {
	if len(x) != lr.features {
		panic("Input vector size does not match the number of features")
	}

	// Add bias term (theta0)
	x = append([]float64{1}, x...)

	// Compute dot product of theta and input vector
	prediction := 0.0
	for i := 0; i <= lr.features; i++ {
		prediction += lr.theta[i] * x[i]
	}

	return prediction
}

// LoadData loads input and output data from a CSV file.
func LoadData(filename string) ([][]float64, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, err
	}

	numFeatures := len(records[0]) - 1
	X := make([][]float64, len(records))
	y := make([]float64, len(records))

	for i, record := range records {
		X[i] = make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			X[i][j], err = strconv.ParseFloat(record[j], 64)
			if err != nil {
				return nil, nil, err
			}
		}
		y[i], err = strconv.ParseFloat(record[numFeatures], 64)
		if err != nil {
			return nil, nil, err
		}
	}

	return X, y, nil
}

// RMSE calculates the root mean squared error between predicted and actual values.
func RMSE(actual, predicted []float64) float64 {
	if len(actual) != len(predicted) {
		panic("Input vector sizes don't match")
	}

	sumSquares := 0.0
	for i := 0; i < len(actual); i++ {
		diff := actual[i] - predicted[i]
		sumSquares += diff * diff
	}

	meanSquaredError := sumSquares / float64(len(actual))
	return math.Sqrt(meanSquaredError)
}

func main() {
	// Load input and output data from a CSV file
	X, y, err := LoadData("data.csv")
	if err != nil {
		log.Fatal(err)
	}

	// Set hyperparameters
	alpha := 0.01        // Learning rate
	numIterations := 100 // Number of iterations

	// Train the linear regression model
	lr := LinearRegression{}
	lr.Fit(X, y, alpha, numIterations)

	// Make predictions for new input vectors
	newX := []float64{1.5, 2.5, 3.5}
	prediction := lr.Predict(newX)
	fmt.Printf("Prediction for input %v: %.2f\n", newX, prediction)

	// Evaluate the model using RMSE
	predictions := make([]float64, len(X))
	for i, input := range X {
		predictions[i] = lr.Predict(input)
	}
	rmse := RMSE(y, predictions)
	fmt.Printf("Root Mean Squared Error: %.2f\n", rmse)
}