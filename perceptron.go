package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

type foodPair struct {
	inputs []float64
	output string
}

func main() {
	trainingSet, _ := os.Open("./diary.txt")
	reader := bufio.NewScanner(trainingSet)

	var trainingPairs []foodPair
	for reader.Scan() {
		pair := parseFoodPair(reader.Text(), true)
		trainingPairs = append(trainingPairs, pair)
	}

	weights := train(trainingPairs)

	testSet, _ := os.Open("./pantry.txt")
	reader = bufio.NewScanner(testSet)
	for reader.Scan() {
		pair := parseFoodPair(reader.Text(), false)
		pair.addBias()
		fmt.Printf("%v: %v\n", pair.inputs, intToTaste(excited(dot(pair, weights))))
	}

	fmt.Printf("\nWeights: %v\n", weights)
}

func train(pairs []foodPair) []float64 {
	learningRate := 0.05
	weights := append([]float64{1.0}, make([]float64, len(pairs[0].inputs))...)

	for _, pair := range pairs {
		pair.addBias()
		res := dot(pair, weights)
		error := tasteToInt(pair) - excited(res)

		weights = updateWeights(learningRate, error, weights, pair.inputs)
	}

	return weights
}

func (pair *foodPair) addBias() {
	pair.inputs = append([]float64{1.0}, pair.inputs...)
}

func updateWeights(lr float64, error int, weights []float64, inputs []float64) []float64 {
	for i, weight := range weights {
		weights[i] = weight + lr*float64(error)*inputs[i]
	}

	return weights
}

func tasteToInt(pair foodPair) int {
	if pair.output == "Yummy!" {
		return 1
	} else {
		return 0
	}
}

func intToTaste(value int) string {
	if value == 0 {
		return "Disgusting!"
	} else {
		return "Yummy!"
	}
}

func parseFoodPair(line string, training bool) foodPair {
	var inputs []string
	arr := strings.Split(line, " ")
	if training == true {
		inputs = arr[:len(arr)-1]
	} else {
		inputs = arr
	}

	output := arr[len(arr)-1]

	var converted []float64
	for _, value := range inputs {
		newValue, _ := strconv.ParseFloat(value, 64)
		converted = append(converted, newValue)
	}

	weights := make([]float64, len(converted))
	for i := 0; i < len(converted); i++ {
		weights[i] = rand.Float64()
	}

	pair := foodPair{inputs: converted, output: output}
	return pair
}

func dot(pair foodPair, weights []float64) float64 {
	sum := 0.0
	for i, input := range pair.inputs {
		sum += input * weights[i]
	}

	return sum
}

func excited(value float64) int {
	if value > 0 {
		return 1
	} else {
		return 0
	}
}
