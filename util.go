package main

import (
	"fmt"
	"log"
	"time"

	"golang.org/x/exp/rand"
	m "gonum.org/v1/gonum/mat"
	s "gonum.org/v1/gonum/stat/distuv"
)

func printMat(mat m.Dense) {
	formatted := m.Formatted(&mat, m.Prefix(""), m.Squeeze())
	fmt.Println(formatted)
}

func colSums(d m.Dense) m.Dense {
	_, c := d.Dims()

	rawSums := make([]float64, c)
	for i := 0; i < c; i++ {
		row := m.DenseCopyOf(d.T()).RawRowView(i)
		for _, v := range row {
			rawSums[i] += v
		}
	}
	return *m.NewDense(1, c, rawSums)
}

func addBiases(inputs m.Dense, biases []float64) m.Dense {
	r, c := inputs.Dims()
	biasMatrix := m.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		biasMatrix.SetRow(i, biases)
	}

	var result m.Dense
	result.Add(&inputs, biasMatrix)

	return result
}

func convertOneHotToLabel(targetOutputs [][]float64) []float64 {
	labels := make([]float64, len(targetOutputs))
	for i, slice := range targetOutputs {
		for j, v := range slice {
			if v == float64(1) {
				labels[i] = float64(j)
			}
		}
	}
	return labels
}

func convertLabelToOneHot(labels []float64) [][]float64 {
	// Initialize the one-hot encoded matrix
	numClasses := len(labels)
	oneHotMatrix := make([][]float64, len(labels))
	for i := range oneHotMatrix {
		oneHotMatrix[i] = make([]float64, numClasses)
	}

	// Fill the matrix with one-hot encodings
	for i, label := range labels {
		if int(label) >= 0 && int(label) < numClasses {
			oneHotMatrix[i][int(label)] = 1.0
		}
	}

	return oneHotMatrix
}

func meanLoss(losses m.Dense) float64 {
	_, c := losses.Dims()
	return m.Sum(&losses) / float64(c)
}

func clipOutputs(inputs m.Dense) m.Dense {
	var outputs m.Dense
	outputs.Apply(func(_, _ int, v float64) float64 {
		if v < 1e-7 {
			return 1e-7
		} else if v > 1+1e-7 {
			return 1 + 1e-7
		}
		return v
	}, &inputs)
	return outputs
}

func accuracy(outputs m.Dense, targetOutputs []float64) float64 {
	r, _ := outputs.Dims()

	rowMaxIndexes := make([]float64, r)
	rowMaxValues := make([]float64, r)
	for i := 0; i < r; i++ {
		row := outputs.RawRowView(i)
		for j, v := range row {
			if v > rowMaxValues[i] {
				rowMaxValues[i] = v
				rowMaxIndexes[i] = float64(j)
			}
		}
	}

	var correct float64
	for i := 0; i < r; i++ {
		if rowMaxIndexes[i] == targetOutputs[i] {
			correct++
		}
	}

	return correct / float64(len(targetOutputs))
}

func binaryAccuracy(outputs m.Dense, targetOutputs []float64) float64 {
    var preds m.Dense
    preds.Apply(func(i, j int, v float64) float64 { if v > 0.5 { return 1 } else { return 0 } }, &outputs)

    var correct float64
    for k, v := range targetOutputs {
        if preds.At(k, 0) == v {
            correct++
        }
    }

    return correct / float64(len(targetOutputs))
}

func bernoulliMat(p float64, r, c int) m.Matrix {
	if r <= 0 || c <= 0 {
		log.Fatal("length of bernoulli matrix too small")
	}

	if p > 1 || p < 0 {
		log.Fatal("probability of bernoulli matrix must be between 0 and 1")
	}

	b := s.Bernoulli{P: p, Src: rand.NewSource(uint64(time.Now().UnixNano()))}
	d := m.NewDense(r, c, nil)
	d.Apply(func(_, _ int, v float64) float64 { return b.Rand() }, d)
	return d
}
