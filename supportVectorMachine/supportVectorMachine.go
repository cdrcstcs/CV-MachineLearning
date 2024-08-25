package supportVectorMachine

import(
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

// SVM represents a Support Vector Machine model
type SVM struct {
	Weights []float64 // Weight vector
	Bias    float64   // Bias term
	C       float64   // Regularization parameter
}

// Train trains the SVM model using the given training data
func (svm *SVM) Train(X [][]float64, y []float64, learningRate float64, epochs int) {
	numFeatures := len(X[0])
	numSamples := len(X)

	// Initialize weights and bias
	svm.Weights = make([]float64, numFeatures)
	for i := range svm.Weights {
		svm.Weights[i] = rand.Float64() // Random initialization
	}
	svm.Bias = rand.Float64() // Random initialization

	// Stochastic Gradient Descent
	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < numSamples; i++ {
			// Compute hinge loss
			prediction := svm.predict(X[i]) // Predict
			hingeLoss := math.Max(0, 1-y[i]*prediction)

			// Update weights and bias
			if hingeLoss != 0 {
				// Update weights
				for j := 0; j < numFeatures; j++ {
					svm.Weights[j] -= learningRate * (svm.C*svm.Weights[j] - y[i]*X[i][j])
				}
				// Update bias
				svm.Bias -= learningRate * y[i]
			}
		}
	}
}

// Predict predicts the class label for a given feature vector
func (svm *SVM) predict(x []float64) float64 {
	activation := svm.Bias
	for i := range x {
		activation += svm.Weights[i] * x[i]
	}
	if activation >= 0 {
		return 1
	}
	return -1
}

// Evaluate evaluates the SVM model on the given test data and returns evaluation metrics
func (svm *SVM) Evaluate(XTest [][]float64, yTest []float64) map[string]float64 {
	accuracy := 0.0
	precision := 0.0
	recall := 0.0
	f1 := 0.0
	numCorrect := 0
	numPositive := 0
	numTruePositive := 0
	for i := range XTest {
		prediction := svm.predict(XTest[i])
		if prediction == yTest[i] {
			numCorrect++
		}
		if prediction == 1 {
			numPositive++
			if yTest[i] == 1 {
				numTruePositive++
			}
		}
	}
	if numPositive != 0 {
		precision = float64(numTruePositive) / float64(numPositive)
	}
	if numTruePositive != 0 {
		recall = float64(numTruePositive) / float64(len(XTest))
	}
	if precision+recall != 0 {
		f1 = 2 * (precision * recall) / (precision + recall)
	}
	accuracy = float64(numCorrect) / float64(len(XTest))

	evaluation := make(map[string]float64)
	evaluation["Accuracy"] = accuracy
	evaluation["Precision"] = precision
	evaluation["Recall"] = recall
	evaluation["F1-score"] = f1

	return evaluation
}

// LoadData loads data from a CSV file
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

	var X [][]float64
	var y []float64

	for _, record := range records {
		var row []float64
		for _, value := range record[:len(record)-1] {
			val, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
			if err != nil {
				return nil, nil, err
			}
			row = append(row, val)
		}
		X = append(X, row)

		label, err := strconv.ParseFloat(strings.TrimSpace(record[len(record)-1]), 64)
		if err != nil {
			return nil, nil, err
		}
		y = append(y, label)
	}

	return X, y, nil
}

// SplitData splits data into training and testing sets
func SplitData(X [][]float64, y []float64, testRatio float64) ([][]float64, [][]float64, []float64, []float64) {
	numTest := int(testRatio * float64(len(X)))

	shuffledIndices := rand.Perm(len(X))
	XShuffled := make([][]float64, len(X))
	yShuffled := make([]float64, len(y))
	for i, index := range shuffledIndices {
		XShuffled[i] = X[index]
		yShuffled[i] = y[index]
	}

	XTrain := XShuffled[numTest:]
	yTrain := yShuffled[numTest:]
	XTest := XShuffled[:numTest]
	yTest := yShuffled[:numTest]

	return XTrain, XTest, yTrain, yTest
}

// main function for demonstration
func main() {
	// Load data
	X, y, err := LoadData("data.csv")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Split data into training and testing sets (80% training, 20% testing)
	XTrain, XTest, yTrain, yTest := SplitData(X, y, 0.2)

	// Initialize SVM model
	svm := SVM{C: 1}

	// Train SVM model
	svm.Train(XTrain, yTrain, 0.01, 1000)

	// Evaluate SVM model
	evaluation := svm.Evaluate(XTest, yTest)

	// Print evaluation metrics
	fmt.Println("Evaluation Metrics:")
	fmt.Println("Accuracy:", evaluation["Accuracy"])
	fmt.Println("Precision:", evaluation["Precision"])
	fmt.Println("Recall:", evaluation["Recall"])
	fmt.Println("F1-score:", evaluation["F1-score"])
}
