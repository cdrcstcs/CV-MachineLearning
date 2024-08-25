package MDPs

import(
	"fmt"
	"math/rand"
	"math"
)

// State represents a state in the MDP
type State int

// Action represents an action in the MDP
type Action int

// MDP represents a Markov Decision Process
type MDP struct {
	NumStates   int
	NumActions  int
	Transitions map[State]map[Action]map[State]float64 // Transition probabilities
	Rewards     map[State]map[Action]float64            // Immediate rewards
}

// NewMDP creates a new MDP
func NewMDP(numStates, numActions int) *MDP {
	return &MDP{
		NumStates:   numStates,
		NumActions:  numActions,
		Transitions: make(map[State]map[Action]map[State]float64),
		Rewards:     make(map[State]map[Action]float64),
	}
}

// AddTransition adds a transition probability
func (mdp *MDP) AddTransition(s State, a Action, sPrime State, prob float64) {
	if mdp.Transitions[s] == nil {
		mdp.Transitions[s] = make(map[Action]map[State]float64)
	}
	if mdp.Transitions[s][a] == nil {
		mdp.Transitions[s][a] = make(map[State]float64)
	}
	mdp.Transitions[s][a][sPrime] = prob
}

// AddReward adds an immediate reward
func (mdp *MDP) AddReward(s State, a Action, reward float64) {
	if mdp.Rewards[s] == nil {
		mdp.Rewards[s] = make(map[Action]float64)
	}
	mdp.Rewards[s][a] = reward
}

// PolicyIteration finds the optimal policy using policy iteration algorithm
func (mdp *MDP) PolicyIteration(gamma float64, epsilon float64) map[State]Action {
	// Initialize arbitrary policy
	policy := make(map[State]Action)
	for s := 0; s < mdp.NumStates; s++ {
		policy[State(s)] = Action(rand.Intn(mdp.NumActions))
	}

	// Iterate until policy converges
	for {
		// Policy Evaluation
		V := make(map[State]float64)
		for s := 0; s < mdp.NumStates; s++ {
			V[State(s)] = 0
		}
		delta := epsilon * 2
		for delta >= epsilon {
			delta = 0
			for s := 0; s < mdp.NumStates; s++ {
				v := V[State(s)]
				newV := 0.0
				for a := 0; a < mdp.NumActions; a++ {
					action := Action(a)
					q := mdp.Rewards[State(s)][action]
					for sPrime, prob := range mdp.Transitions[State(s)][action] {
						q += gamma * prob * V[sPrime]
					}
					if a == 0 || q > newV {
						newV = q
					}
				}
				V[State(s)] = newV
				delta = math.Max(delta, math.Abs(v-newV))
			}
		}

		// Policy Improvement
		policyStable := true
		for s := 0; s < mdp.NumStates; s++ {
			oldAction := policy[State(s)]
			maxAction := Action(0)
			maxQ := -1e9
			for a := 0; a < mdp.NumActions; a++ {
				action := Action(a)
				q := mdp.Rewards[State(s)][action]
				for sPrime, prob := range mdp.Transitions[State(s)][action] {
					q += gamma * prob * V[sPrime]
				}
				if q > maxQ {
					maxQ = q
					maxAction = action
				}
			}
			policy[State(s)] = maxAction
			if oldAction != maxAction {
				policyStable = false
			}
		}

		if policyStable {
			break
		}
	}

	return policy
}

func main() {
	// Create a simple MDP
	mdp := NewMDP(3, 2)

	// Define transitions
	mdp.AddTransition(0, 0, 1, 0.5)
	mdp.AddTransition(0, 1, 2, 1.0)
	mdp.AddTransition(1, 0, 0, 0.8)
	mdp.AddTransition(1, 1, 2, 0.4)
	mdp.AddTransition(2, 0, 0, 0.2)
	mdp.AddTransition(2, 1, 1, 0.6)

	// Define rewards
	mdp.AddReward(0, 0, 0.5)
	mdp.AddReward(0, 1, 1.0)
	mdp.AddReward(1, 0, -0.2)
	mdp.AddReward(1, 1, 0.8)
	mdp.AddReward(2, 0, 0.1)
	mdp.AddReward(2, 1, -0.5)

	// Perform policy iteration to find optimal policy
	optimalPolicy := mdp.PolicyIteration(0.9, 0.01)

	// Print optimal policy
	fmt.Println("Optimal Policy:")
	for state, action := range optimalPolicy {
		fmt.Printf("State %d: Action %d\n", state, action)
	}
}
