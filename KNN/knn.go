package KNN

import(
	"fmt"
	"math"
)

type DataPoint struct {
	Features []float64
	Label    string
}

func euclideanDistance(p1, p2 []float64) float64 {
	sum := 0.0
	for i := range p1 {
		diff := p1[i] - p2[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func findKNearestNeighbors(data []DataPoint, query []float64, k int) []string {
	distances := make([]float64, len(data))
	for i, point := range data {
		distances[i] = euclideanDistance(point.Features, query)
	}

	// Sort indices based on distances
	sortedIndices := make([]int, len(data))
	for i := range sortedIndices {
		sortedIndices[i] = i
	}
	for i := range distances {
		for j := range distances[:i] {
			if distances[i] < distances[j] {
				distances[i], distances[j] = distances[j], distances[i]
				sortedIndices[i], sortedIndices[j] = sortedIndices[j], sortedIndices[i]
			}
		}
	}

	// Get the labels of the k nearest neighbors
	nearestLabels := make([]string, k)
	for i := 0; i < k; i++ {
		nearestLabels[i] = data[sortedIndices[i]].Label
	}

	return nearestLabels
}

func main() {
	// Sample dataset
	data := []DataPoint{
		{Features: []float64{5.1, 3.5}, Label: "A"},
		{Features: []float64{4.9, 3.0}, Label: "A"},
		{Features: []float64{7.0, 3.2}, Label: "B"},
		{Features: []float64{6.4, 3.2}, Label: "B"},
	}

	// Query point
	query := []float64{6.0, 3.0}

	// Find k nearest neighbors
	k := 2
	nearestLabels := findKNearestNeighbors(data, query, k)

	fmt.Printf("Query point belongs to labels: %v\n", nearestLabels)
}
