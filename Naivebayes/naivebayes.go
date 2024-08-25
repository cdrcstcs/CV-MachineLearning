package Naivebayes

import(
    "fmt"
    "math"
)

// NaiveBayes represents the Naive Bayes classifier.
type NaiveBayes struct {
    classCounts map[string]int
    wordCounts  map[string]map[string]int
}

// NewNaiveBayes initializes a new NaiveBayes classifier.
func NewNaiveBayes() *NaiveBayes {
    return &NaiveBayes{
        classCounts: make(map[string]int),
        wordCounts:  make(map[string]map[string]int),
    }
}

// Train trains the NaiveBayes classifier with the given data.
func (nb *NaiveBayes) Train(data [][]string, labels []string) {
    for i := range data {
        label := labels[i]
        nb.classCounts[label]++
        if nb.wordCounts[label] == nil {
            nb.wordCounts[label] = make(map[string]int)
        }
        for _, word := range data[i] {
            nb.wordCounts[label][word]++
        }
    }
}

// Predict predicts the class label for the given input.
func (nb *NaiveBayes) Predict(input []string) string {
    var bestLabel string
    var bestProb = -math.MaxFloat64

    for label := range nb.classCounts {
        prob := nb.calculateClassProbability(input, label)
        if prob > bestProb {
            bestProb = prob
            bestLabel = label
        }
    }
    return bestLabel
}

// calculateClassProbability calculates the probability of the given input belonging to the specified class.
func (nb *NaiveBayes) calculateClassProbability(input []string, label string) float64 {
    prob := math.Log(float64(nb.classCounts[label]) / float64(len(nb.classCounts)))
    for _, word := range input {
        if nb.wordCounts[label][word] > 0 {
            prob += math.Log(float64(nb.wordCounts[label][word]) / float64(nb.classCounts[label]))
        }
    }
    return prob
}

func main() {
    // Create a new NaiveBayes classifier
    nb := NewNaiveBayes()

    // Sample training data
    data := [][]string{
        {"free", "money"},
        {"meeting", "tomorrow"},
        {"click", "here", "to", "win"},
    }
    labels := []string{"spam", "ham", "spam"}

    // Train the classifier
    nb.Train(data, labels)

    // Sample input for prediction
    input := []string{"free", "money"}

    // Predict the class label
    predictedLabel := nb.Predict(input)
    fmt.Println("Predicted label:", predictedLabel)
}
