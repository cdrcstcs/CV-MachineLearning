package hyperparameterTuning

import(
	"math"
	"math/rand"
)

// Model represents a machine learning model.
type Model interface {
	Fit(X [][]float64, y []float64)
	Predict(X []float64) float64
	SetParameter(param string, value float64)
}

// EvaluationFunction is a function type for evaluating model performance.
type EvaluationFunction func(yTrue, yPred []float64) float64

// HyperparameterTuningResult represents the result of hyperparameter tuning.
type HyperparameterTuningResult struct {
	BestParams map[string]float64
	BestScore  float64
}

// GridSearch performs hyperparameter tuning using grid search.
func GridSearch(model Model, paramGrid map[string][]float64, evalFunc EvaluationFunction, X [][]float64, y []float64, numFolds int) (*HyperparameterTuningResult, error) {
	bestScore := math.Inf(-1)
	bestParams := make(map[string]float64)

	// Generate all combinations of parameters
	paramCombos := parameterCombinations(paramGrid)

	// Iterate over parameter combinations
	for _, params := range paramCombos {
		// Set model parameters
		for param, value := range params {
			model.SetParameter(param, value)
		}

		// Perform cross-validation
		scores := make([]float64, numFolds)
		for i := 0; i < numFolds; i++ {
			XTrain, yTrain, XValid, yValid := splitData(X, y, 1.0/float64(numFolds))
			model.Fit(XTrain, yTrain)
			yPred := make([]float64, len(XValid))
			for j, sample := range XValid {
				yPred[j] = model.Predict(sample)
			}
			scores[i] = evalFunc(yValid, yPred)
		}

		// Compute average score
		avgScore := average(scores)

		// Update best parameters if necessary
		if avgScore > bestScore {
			bestScore = avgScore
			for param, value := range params {
				bestParams[param] = value
			}
		}
	}

	return &HyperparameterTuningResult{
		BestParams: bestParams,
		BestScore:  bestScore,
	}, nil
}
// splitData splits the data into training and validation sets
func splitData(X [][]float64, y []float64, splitRatio float64) ([][]float64, []float64, [][]float64, []float64) {
    // Calculate the number of samples for the training set
    numTrain := int(float64(len(X)) * splitRatio)

    // Split the features into training and validation sets
    XTrain := X[:numTrain]
    XValid := X[numTrain:]

    // Split the target values into training and validation sets
    yTrain := y[:numTrain]
    yValid := y[numTrain:]

    return XTrain, yTrain, XValid, yValid
}

// RandomizedSearch performs hyperparameter tuning using randomized search.
func RandomizedSearch(model Model, paramGrid map[string][]float64, evalFunc EvaluationFunction, X [][]float64, y []float64, numIterations int) (*HyperparameterTuningResult, error) {
	bestScore := math.Inf(-1)
	bestParams := make(map[string]float64)

	// Iterate over random parameter combinations
	for i := 0; i < numIterations; i++ {
		// Generate random parameters
		params := randomParameters(paramGrid)

		// Set model parameters
		for param, value := range params {
			model.SetParameter(param, value)
		}

		// Perform cross-validation
		XTrain, yTrain, XValid, yValid := splitData(X, y, 0.8)
		model.Fit(XTrain, yTrain)
		yPred := make([]float64, len(XValid))
		for j, sample := range XValid {
			yPred[j] = model.Predict(sample)
		}
		score := evalFunc(yValid, yPred)

		// Update best parameters if necessary
		if score > bestScore {
			bestScore = score
			for param, value := range params {
				bestParams[param] = value
			}
		}
	}

	return &HyperparameterTuningResult{
		BestParams: bestParams,
		BestScore:  bestScore,
	}, nil
}

// parameterCombinations generates all combinations of parameters from the parameter grid.
func parameterCombinations(paramGrid map[string][]float64) []map[string]float64 {
	var keys []string
	for key := range paramGrid {
		keys = append(keys, key)
	}
	return parameterCombinationsHelper(keys, paramGrid, make(map[string]float64), nil)
}

func parameterCombinationsHelper(keys []string, paramGrid map[string][]float64, params map[string]float64, result []map[string]float64) []map[string]float64 {
	if len(keys) == 0 {
		// Base case: all parameters are set
		paramCopy := make(map[string]float64)
		for k, v := range params {
			paramCopy[k] = v
		}
		result = append(result, paramCopy)
		return result
	}
	// Recursive case: set one parameter and recurse
	key := keys[0]
	values := paramGrid[key]
	for _, value := range values {
		params[key] = value
		result = parameterCombinationsHelper(keys[1:], paramGrid, params, result)
	}
	return result
}

// randomParameters generates random parameters from the parameter grid.
func randomParameters(paramGrid map[string][]float64) map[string]float64 {
	params := make(map[string]float64)
	for param, values := range paramGrid {
		params[param] = values[rand.Intn(len(values))]
	}
	return params
}

// average calculates the average of a slice of float64 values.
func average(arr []float64) float64 {
	sum := 0.0
	for _, value := range arr {
		sum += value
	}
	return sum / float64(len(arr))
}

// SplitData splits the data into training and validation sets.
func SplitData(X [][]float64, y []float64, splitRatio float64) ([][]float64, []float64, [][]float64, []float64) {
	numTrain := int(float64(len(X)) * splitRatio)
	XTrain := X[:numTrain]
	yTrain := y[:numTrain]
	XValid := X[numTrain:]
	yValid := y[numTrain:]
	return XTrain, yTrain, XValid, yValid
}
