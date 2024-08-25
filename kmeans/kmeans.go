package kmeans

import(
	"fmt"
	"math"
	"math/rand"
)

// Point represents a data point in a multidimensional space
type Point struct {
	Values []float64
}

// Cluster represents a cluster of data points
type Cluster struct {
	Centroid Point
	Points   []Point
}

// KMeans performs k-means clustering on a given dataset
func KMeans(data []Point, k int, maxIterations int) ([]Cluster, error) {
	if len(data) < k {
		return nil, fmt.Errorf("not enough data points for %d clusters", k)
	}

	// Initialize random centroids
	centroids := getRandomCentroids(data, k)

	// Create initial clusters
	clusters := make([]Cluster, k)
	for i := range clusters {
		clusters[i].Centroid = centroids[i]
	}

	// Run k-means iterations
	for iteration := 0; iteration < maxIterations; iteration++ {
		// Assign data points to clusters
		for _, point := range data {
			closestClusterIndex := getClosestClusterIndex(point, clusters)
			clusters[closestClusterIndex].Points = append(clusters[closestClusterIndex].Points, point)
		}

		// Update centroids of clusters
		for i := range clusters {
			if len(clusters[i].Points) > 0 {
				clusters[i].Centroid = calculateCentroid(clusters[i].Points)
			}
		}

		// Clear points from clusters for the next iteration
		for i := range clusters {
			clusters[i].Points = nil
		}
	}

	return clusters, nil
}

// getRandomCentroids returns random centroids from the given data
func getRandomCentroids(data []Point, k int) []Point {
	rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })
	return data[:k]
}

// getClosestClusterIndex returns the index of the closest cluster to a given point
func getClosestClusterIndex(point Point, clusters []Cluster) int {
	minDistance := math.Inf(1)
	closestIndex := 0

	for i, cluster := range clusters {
		distance := euclideanDistance(point, cluster.Centroid)
		if distance < minDistance {
			minDistance = distance
			closestIndex = i
		}
	}

	return closestIndex
}

// calculateCentroid calculates the centroid of a cluster
func calculateCentroid(points []Point) Point {
	if len(points) == 0 {
		return Point{}
	}

	dimension := len(points[0].Values)
	sumValues := make([]float64, dimension)
	for _, point := range points {
		for i := range point.Values {
			sumValues[i] += point.Values[i]
		}
	}

	centroidValues := make([]float64, dimension)
	for i := range centroidValues {
		centroidValues[i] = sumValues[i] / float64(len(points))
	}

	return Point{Values: centroidValues}
}

// euclideanDistance calculates the Euclidean distance between two points
func euclideanDistance(a Point, b Point) float64 {
	if len(a.Values) != len(b.Values) {
		return math.Inf(1)
	}

	sum := 0.0
	for i := range a.Values {
		diff := a.Values[i] - b.Values[i]
		sum += diff * diff
	}

	return math.Sqrt(sum)
}

func main() {
	// Sample data points in 2-dimensional space
	data := []Point{
		{Values: []float64{2, 3}},
		{Values: []float64{4, 5}},
		{Values: []float64{6, 7}},
		{Values: []float64{8, 9}},
		{Values: []float64{10, 11}},
		{Values: []float64{12, 13}},
		{Values: []float64{14, 15}},
		{Values: []float64{16, 17}},
	}

	k := 2       // Number of clusters
	maxIter := 10 // Maximum iterations for k-means

	clusters, err := KMeans(data, k, maxIter)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Print the clusters and their centroids
	for i, cluster := range clusters {
		fmt.Printf("Cluster %d:\n", i+1)
		fmt.Println("Centroid:", cluster.Centroid)
		fmt.Println("Points:", cluster.Points)
	}
}
