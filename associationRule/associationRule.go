package associationRule

import(
	"fmt"
	"sort"
	"strings"
)

// Itemset represents a set of items
type Itemset []string

// Equal checks if two Itemsets are equal
func (s Itemset) Equal(other Itemset) bool {
	if len(s) != len(other) {
		return false
	}
	for i, item := range s {
		if item != other[i] {
			return false
		}
	}
	return true
}

// Hash calculates the hash value for an Itemset
func (s Itemset) Hash() string {
	return strings.Join(s, ",")
}

// Transaction represents a transaction in the dataset
type Transaction []string

// AssociationRule represents an association rule
type AssociationRule struct {
	Antecedent Itemset
	Consequent Itemset
	Support    float64
	Confidence float64
	Lift       float64
}

// AssociationRuleSet represents a set of association rules
type AssociationRuleSet []AssociationRule

// GenerateAssociationRules generates association rules from the given transactions
func GenerateAssociationRules(transactions []Transaction, minSupport, minConfidence float64) AssociationRuleSet {
	// Step 1: Find frequent itemsets
	frequentItemsets := findFrequentItemsets(transactions, minSupport)

	// Step 2: Generate association rules from frequent itemsets
	rules := make(AssociationRuleSet, 0)
	for _, itemset := range frequentItemsets {
		if len(itemset) > 1 {
			subsets := generateSubsets(itemset)
			for _, subset := range subsets {
				antecedent := subset
				consequent := getDifference(itemset, subset)
				rule := AssociationRule{
					Antecedent: antecedent,
					Consequent: consequent,
					Support:    calculateSupport(itemset, transactions),
				}
				if rule.Support >= minSupport {
					rule.Confidence = calculateConfidence(antecedent, consequent, transactions)
					if rule.Confidence >= minConfidence {
						rule.Lift = calculateLift(rule.Confidence, calculateSupport(consequent, transactions))
						rules = append(rules, rule)
					}
				}
			}
		}
	}
	return rules
}

// findFrequentItemsets finds frequent itemsets from transactions using Apriori algorithm
func findFrequentItemsets(transactions []Transaction, minSupport float64) []Itemset {
	frequentItemsets := make([]Itemset, 0)
	itemsetCount := make(map[string]int)
	candidates := generateInitialCandidates(transactions)

	for _, transaction := range transactions {
		for _, candidate := range candidates {
			if containsItem(transaction, candidate) {
				itemsetCount[candidate.Hash()]++
			}
		}
	}

	for itemsetStr, count := range itemsetCount {
		itemset := strings.Split(itemsetStr, ",")
		support := float64(count) / float64(len(transactions))
		if support >= minSupport {
			frequentItemsets = append(frequentItemsets, itemset)
		}
	}
	return frequentItemsets
}

// generateInitialCandidates generates initial candidates from transactions
func generateInitialCandidates(transactions []Transaction) []Itemset {
	candidates := make([]Itemset, 0)
	itemSet := make(map[string]bool)

	for _, transaction := range transactions {
		for _, item := range transaction {
			itemSet[item] = true
		}
	}

	for item := range itemSet {
		candidates = append(candidates, Itemset{item})
	}
	return candidates
}

// containsItem checks if an itemset contains all items in a transaction
func containsItem(transaction Transaction, itemset Itemset) bool {
	for _, item := range itemset {
		found := false
		for _, tItem := range transaction {
			if item == tItem {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// generateSubsets generates all possible subsets of an itemset
func generateSubsets(itemset Itemset) []Itemset {
	subsets := make([]Itemset, 0)
	generateSubsetsHelper(itemset, 0, &[]string{}, &subsets)
	return subsets
}

// generateSubsetsHelper is a helper function for generating subsets recursively
func generateSubsetsHelper(itemset Itemset, index int, current *[]string, subsets *[]Itemset) {
	if index == len(itemset) {
		*subsets = append(*subsets, *current)
		return
	}
	*current = append(*current, itemset[index])
	generateSubsetsHelper(itemset, index+1, current, subsets)
	*current = (*current)[:len(*current)-1]
	generateSubsetsHelper(itemset, index+1, current, subsets)
}

// getDifference returns the difference of two itemsets
func getDifference(itemset, subset Itemset) Itemset {
	difference := make(Itemset, 0)
	for _, item := range itemset {
		found := false
		for _, sItem := range subset {
			if item == sItem {
				found = true
				break
			}
		}
		if !found {
			difference = append(difference, item)
		}
	}
	return difference
}

// calculateSupport calculates the support of an itemset in transactions
func calculateSupport(itemset Itemset, transactions []Transaction) float64 {
	count := 0
	for _, transaction := range transactions {
		if containsItem(transaction, itemset) {
			count++
		}
	}
	return float64(count) / float64(len(transactions))
}

// calculateConfidence calculates the confidence of a rule
func calculateConfidence(antecedent, consequent Itemset, transactions []Transaction) float64 {
	combined := append(antecedent, consequent...)
	return calculateSupport(combined, transactions) / calculateSupport(antecedent, transactions)
}

// calculateLift calculates the lift of a rule
func calculateLift(confidence, consequentSupport float64) float64 {
	return confidence / consequentSupport
}

func main() {
	// Sample transactions
	transactions := []Transaction{
		{"bread", "milk"},
		{"bread", "diaper", "beer", "egg"},
		{"milk", "diaper", "beer", "cola"},
		{"bread", "milk", "diaper", "beer"},
		{"bread", "milk", "diaper", "cola"},
	}

	// Minimum support and confidence thresholds
	minSupport := 0.4
	minConfidence := 0.6

	// Generate association rules
	rules := GenerateAssociationRules(transactions, minSupport, minConfidence)

	// Sort rules by lift
	sort.Slice(rules, func(i, j int) bool {
		return rules[i].Lift > rules[j].Lift
	})

	// Print association rules
	fmt.Println("Association Rules:")
	for _, rule := range rules {
		fmt.Printf("%v -> %v (Support: %.2f, Confidence: %.2f, Lift: %.2f)\n", rule.Antecedent, rule.Consequent, rule.Support, rule.Confidence, rule.Lift)
	}
}
