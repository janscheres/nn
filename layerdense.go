package main

import (
	"math/rand"

	m "gonum.org/v1/gonum/mat"
)

type layerDense struct {
	weights m.Dense
	biases  []float64
	output  m.Dense

	// for backwards propogation
	inputs   m.Dense
	dWeights m.Dense
	dBiases  m.Dense
	dInputs  m.Dense

	// for gradient descents (optimizers)
	weightMomentums m.Dense
	biasMomentums   []float64
	weightCache     m.Dense
	biasCache       []float64

	// for regularization, dictates how much of an impact regularization penalty should have
	wRegularizerL1 float64
	wRegularizerL2 float64
	bRegularizerL1 float64
	bRegularizerL2 float64
}

func newLayerDense(nInputs, nNeurons int, wRegularizerL1, wRegularizerL2, bRegularizerL1, bRegularizerL2 float64) layerDense {
	// fill weights with random data
	data := make([]float64, nInputs*nNeurons)
	for i := range data {
		data[i] = rand.NormFloat64() * 0.01
	}

	weights := m.NewDense(nInputs, nNeurons, data)

	biases := make([]float64, nNeurons)

	return layerDense{weights: *weights, biases: biases, wRegularizerL1: wRegularizerL1, wRegularizerL2: wRegularizerL2, bRegularizerL1: bRegularizerL1, bRegularizerL2: bRegularizerL2}
}

func (l *layerDense) Forward(inputs m.Dense) {
	l.inputs = inputs

	var dotProd m.Dense
	dotProd.Mul(&inputs, &l.weights)

	l.output = addBiases(dotProd, l.biases)
}

func (l *layerDense) Backward(dVals m.Dense) {
	l.dWeights.Mul(l.inputs.T(), &dVals)
	l.dBiases = colSums(dVals)

	if l.wRegularizerL1 > 0 {
		var wdL1 m.Dense
		wdL1.Apply(func(_, _ int, v float64) float64 {
			if v >= 0 {
				return l.wRegularizerL1
			} else {
				return -l.wRegularizerL1
			}
		}, &l.weights)
		l.dWeights.Add(&l.dWeights, &wdL1)
	}

	if l.wRegularizerL2 > 0 {
		var wdL2 m.Dense
		wdL2.Apply(func(_, _ int, v float64) float64 {
			return 2 * l.wRegularizerL2 * v
		}, &l.weights)
		l.dWeights.Add(&l.dWeights, &wdL2)
	}

	if l.bRegularizerL1 > 0 {
		bdL1 := m.NewDense(1, len(l.biases), nil)
		for k, v := range l.biases {
			if v >= 0 {
				bdL1.Set(0, k, l.bRegularizerL1)
			} else {
				bdL1.Set(0, k, -l.bRegularizerL1)
			}
		}
		l.dBiases.Add(&l.dBiases, bdL1)
	}

	if l.bRegularizerL2 > 0 {
		bdL2 := m.NewDense(1, len(l.biases), nil)
		for k, v := range l.biases {
			bdL2.Set(0, k, 2*l.bRegularizerL2*v)
		}
		l.dBiases.Add(&l.dBiases, bdL2)
	}

	l.dInputs.Mul(&dVals, l.weights.T())
}
