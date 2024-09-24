package main

import (
	m "gonum.org/v1/gonum/mat"
)

type dropOutLayer struct {
	inputs     m.Dense
	output     m.Dense
	binaryMask m.Dense
	dInputs    m.Dense
	rate       float64
}

func newDropOutLater(r float64) dropOutLayer {
	return dropOutLayer{rate: 1 - r}
}

func (l *dropOutLayer) Forward(inputs m.Dense) {
	l.inputs = inputs
	r, c := inputs.Dims()
	l.binaryMask.Apply(func(_, _ int, v float64) float64 { return v / l.rate }, bernoulliMat(l.rate, r, c))
	l.output.Apply(func(i, j int, v float64) float64 { return v * l.binaryMask.At(i, j) }, &inputs)
}

func (l *dropOutLayer) Backward(dVals m.Dense) {
	l.dInputs.Apply(func(i, j int, v float64) float64 { return v * l.binaryMask.At(i, j) }, &dVals)
}
