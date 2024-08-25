package hierachicalCLustering

import(
	"fmt"
	"math"
)

type Cluster struct {
	Points [][]float64
	Center []float64
}

func distance(p1, p2 []float64) float64 {
	sum := 0.0
	for i := range p1 {
		diff := p1[i] - p2[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func centroid(points [][]float64) []float64 {
	if len(points) == 0 {
		return nil
	}
	dim := len(points[0])
	center := make([]float64, dim)
	for _, point := range points {
		for i, coord := range point {
			center[i] += coord
		}
	}
	for i := range center {
		center[i] /= float64(len(points))
	}
	return center
}

func agglomerativeClustering(data [][]float64, k int) [][]int {
	clusters := make([]Cluster, len(data))
	for i := range clusters {
		clusters[i].Points = [][]float64{data[i]}
		clusters[i].Center = data[i]
	}

	for len(clusters) > k {
		minDistance := math.Inf(1)
		mergeIdx1, mergeIdx2 := -1, -1
		for i := 0; i < len(clusters); i++ {
			for j := i + 1; j < len(clusters); j++ {
				d := distance(clusters[i].Center, clusters[j].Center)
				if d < minDistance {
					minDistance = d
					mergeIdx1, mergeIdx2 = i, j
				}
			}
		}

		newCluster := Cluster{
			Points: append(clusters[mergeIdx1].Points, clusters[mergeIdx2].Points...),
			Center: centroid(append(clusters[mergeIdx1].Points, clusters[mergeIdx2].Points...)),
		}

		// Remove the clusters being merged and add the new cluster
		clusters = append(clusters[:mergeIdx2], clusters[mergeIdx2+1:]...)
		clusters = append(clusters[:mergeIdx1], clusters[mergeIdx1+1:]...)
		clusters = append(clusters, newCluster)
	}

	// Convert clusters to cluster assignments
	assignments := make([][]int, len(data))
	for i := range data {
		for j, cluster := range clusters {
			for _, point := range cluster.Points {
				if equalPoints(data[i], point) {
					assignments[i] = append(assignments[i], j)
				}
			}
		}
	}
	return assignments
}

func equalPoints(p1, p2 []float64) bool {
	if len(p1) != len(p2) {
		return false
	}
	for i := range p1 {
		if p1[i] != p2[i] {
			return false
		}
	}
	return true
}

func main() {
	data := [][]float64{
		{1, 1},
		{2, 2},
		{10, 10},
		{11, 11},
	}

	k := 2
	assignments := agglomerativeClustering(data, k)

	fmt.Println("Cluster Assignments:")
	for i, assignment := range assignments {
		fmt.Printf("Data point %d belongs to cluster(s): %v\n", i, assignment)
	}
}
