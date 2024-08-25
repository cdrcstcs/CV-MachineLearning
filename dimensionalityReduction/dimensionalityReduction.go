package dimensionalityReduction

import (
	"fmt"
	"math"
	"sort"
)

// PCA struct holds the Principal Component Analysis parameters
type PCA struct {
	Components         int       // Number of principal components
	Mean               []float64 // Mean of each feature
	Vectors            [][]float64 // Principal components
	ExplainedVariance  []float64 // Explained variance
	ExplainedVarianceRatio  []float64 // Explained variance ratio
}

// Fit method computes the mean and principal components of the input data
func (p *PCA) Fit(data [][]float64) {
	rows := len(data)
	cols := len(data[0])

	// Compute mean of each feature
	p.Mean = make([]float64, cols)
	for i := 0; i < cols; i++ {
		sum := 0.0
		for j := 0; j < rows; j++ {
			sum += data[j][i]
		}
		p.Mean[i] = sum / float64(rows)
	}

	// Subtract mean from data
	centered := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		centered[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			centered[i][j] = data[i][j] - p.Mean[j]
		}
	}

	// Compute covariance matrix
	var covariance [][]float64
	covariance = make([][]float64, cols)
	for i := range covariance {
		covariance[i] = make([]float64, cols)
	}
	for i := 0; i < cols; i++ {
		for j := 0; j < cols; j++ {
			sum := 0.0
			for k := 0; k < rows; k++ {
				sum += centered[k][i] * centered[k][j]
			}
			covariance[i][j] = sum / float64(rows-1)
		}
	}

	// Compute eigenvectors and eigenvalues of covariance matrix
	values, vectors := eigen(covariance)

	// Sort eigenvectors by eigenvalues
	sortEigen := func(eigenvalues []float64, eigenvectors [][]float64) {
		sortedIndices := make([]int, len(eigenvalues))
		for i := range sortedIndices {
			sortedIndices[i] = i
		}
		sortByEigen := func(i, j int) bool { return eigenvalues[i] > eigenvalues[j] }
		sort.Slice(sortedIndices, sortByEigen)

		for i := 0; i < len(eigenvectors); i++ {
			tempCol := make([]float64, len(eigenvectors[0]))
			for j := range sortedIndices {
				tempCol[j] = eigenvectors[sortedIndices[j]][i]
			}
			for j := range tempCol {
				eigenvectors[j][i] = tempCol[j]
			}
		}
	}
	sortEigen(values, vectors)

	// Select only the top Components eigenvectors
	p.Vectors = vectors[:p.Components]

	// Compute explained variance
	p.ExplainedVariance = make([]float64, p.Components)
	for i := 0; i < p.Components; i++ {
		p.ExplainedVariance[i] = values[i]
	}

	// Compute explained variance ratio
	totalVariance := 0.0
	for _, val := range values {
		totalVariance += val
	}
	p.ExplainedVarianceRatio = make([]float64, p.Components)
	for i := 0; i < p.Components; i++ {
		p.ExplainedVarianceRatio[i] = p.ExplainedVariance[i] / totalVariance
	}
}

// Transform method projects the input data onto the principal components
func (p *PCA) Transform(data [][]float64) [][]float64 {
	rows := len(data)
	cols := len(data[0])

	// Subtract mean from data
	centered := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		centered[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			centered[i][j] = data[i][j] - p.Mean[j]
		}
	}

	// Project data onto principal components
	var transformed [][]float64
	transformed = make([][]float64, rows)
	for i := range transformed {
		transformed[i] = make([]float64, p.Components)
		for j := 0; j < p.Components; j++ {
			sum := 0.0
			for k := 0; k < cols; k++ {
				sum += centered[i][k] * p.Vectors[k][j]
			}
			transformed[i][j] = sum
		}
	}
	return transformed
}

// eigen computes the eigenvalues and eigenvectors of a symmetric matrix
func eigen(matrix [][]float64) (values []float64, vectors [][]float64) {
	cols := len(matrix[0])

	// Initialize eigenvectors matrix
	vectors = make([][]float64, cols)
	for i := range vectors {
		vectors[i] = make([]float64, cols)
	}

	// Initialize values
	values = make([]float64, cols)

	// Initialize temp matrix
	temp := make([][]float64, cols)
	for i := range temp {
		temp[i] = make([]float64, cols)
		copy(temp[i], matrix[i])
	}

	for i := 0; i < 1000; i++ { // Max iterations
		// Find max off-diagonal element
		p := 0
		q := 1
		maxVal := math.Abs(temp[0][1])
		for j := 0; j < cols; j++ {
			for k := j + 1; k < cols; k++ {
				if math.Abs(temp[j][k]) > maxVal {
					maxVal = math.Abs(temp[j][k])
					p = j
					q = k
				}
			}
		}

		// Check convergence
		if maxVal < 1e-10 {
			break
		}

		// Compute rotation angle
		theta := 0.5 * math.Atan2(2*temp[p][q], temp[q][q]-temp[p][p])

		// Construct rotation matrix
		c := math.Cos(theta)
		s := math.Sin(theta)
		rot := make([][]float64, cols)
		for j := range rot {
			rot[j] = make([]float64, cols)
			for k := range rot[j] {
				if j == p && k == p || j == q && k == q {
					rot[j][k] = c
				} else if j == p && k == q {
					rot[j][k] = s
				} else if j == q && k == p {
					rot[j][k] = -s
				} else {
					rot[j][k] = 0
				}
			}
		}

		// Apply rotation
		rotT := transpose(rot)
		temp = matmul(rotT, matmul(temp, rot))

		// Update eigenvectors
		vectors = matmul(vectors, rot)
	}

	// Extract eigenvalues
	for i := 0; i < cols; i++ {
		values[i] = temp[i][i]
	}

	return values, vectors
}

// transpose computes the transpose of a matrix
func transpose(matrix [][]float64) [][]float64 {
	rows := len(matrix)
	cols := len(matrix[0])

	transposed := make([][]float64, cols)
	for i := range transposed {
		transposed[i] = make([]float64, rows)
		for j := range transposed[i] {
			transposed[i][j] = matrix[j][i]
		}
	}

	return transposed
}

// matmul computes the matrix multiplication of two matrices
func matmul(a, b [][]float64) [][]float64 {
	rowsA := len(a)
	colsA := len(a[0])
	colsB := len(b[0])

	result := make([][]float64, rowsA)
	for i := range result {
		result[i] = make([]float64, colsB)
	}

	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			for k := 0; k < colsA; k++ {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}

	return result
}

func main() {
	// Sample data
	rawData := [][]float64{
		{1, 2},
		{2, 3},
		{3, 4},
		{4, 5},
		{5, 6},
	}

	// Initialize PCA with 1 component
	pca := &PCA{Components: 1}

	// Fit PCA
	pca.Fit(rawData)

	// Transform data
	transformed := pca.Transform(rawData)

	// Print transformed data
	fmt.Println("Transformed Data:")
	for _, row := range transformed {
		fmt.Println(row)
	}

	// Print explained variance
	fmt.Println("Explained Variance:", pca.ExplainedVariance)

	// Print explained variance ratio
	fmt.Println("Explained Variance Ratio:", pca.ExplainedVarianceRatio)
}
