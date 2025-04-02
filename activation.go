package main

import (
	"math"

	m "gonum.org/v1/gonum/mat"
)

type activationFunc struct {
	inputs  m.Dense
	output  m.Dense
	dInputs m.Dense

	Forward  func(m.Dense)
	Backward func(m.Dense)
}

/*func (a *activationFunc) Forward(inputs m.Dense) {
	a.inputs = inputs
	a.output.Apply(a.forwardFunc, &inputs)
}

func (a *activationFunc) Backward(dVals m.Dense) {
	a.dInputs.Apply(a.backwardFunc, &dVals)
}*/

// recitfied linear units
func reluActivation() *activationFunc {
	act := &activationFunc{}

	act.Forward = func(inputs m.Dense) {
		act.inputs = inputs
		act.output.Apply(func(_, _ int, v float64) float64 {
			if v > 0 {
				return v
			}
			return 0
		}, &inputs)
	}

	act.Backward = func(dVals m.Dense) {
		act.dInputs.Apply(func(i, j int, v float64) float64 {
			if act.inputs.At(i, j) <= 0 {
				return 0
			}
			return v
		}, &dVals)
	}

	return act
}

func softMaxActivation() *activationFunc {
	act := &activationFunc{}

	act.Forward = func(inputs m.Dense) {
		act.inputs = inputs
		r, _ := act.inputs.Dims()

		// get row maxima to then subtract to avoid overflow
		rowMax := make([]float64, r)
		for i := 0; i < r; i++ {
			row := act.inputs.RawRowView(i)
			for _, v := range row {
				if v > rowMax[i] {
					rowMax[i] = v
				}
			}
		}

		// subtract row maxima then exponentiate to remove negatives
		var expVals m.Dense
		rowSums := make([]float64, r)

		expVals.Apply(func(i, _ int, v float64) float64 {
			s := v - rowMax[i]
			value := math.Exp(s)
			rowSums[i] += value
			return value
		}, &act.inputs)

		var norm m.Dense
		norm.Apply(func(i, _ int, v float64) float64 {
			return v / rowSums[i]
		}, &expVals)

		act.output = norm
	}

	act.Backward = func(dVals m.Dense) {
		r, c := act.output.Dims()
		dInputs := m.NewDense(r, c, nil)

		for i := 0; i < r; i++ {
			singleOutput := act.output.RawRowView(i)
			singleDVals := dVals.RawRowView(i)

			//reshape output to column vector
			outputDense := m.NewDense(len(singleOutput), 1, singleOutput)

			var softMaxMul m.Dense
			softMaxMul.Mul(outputDense, outputDense.T())

			var jacobian m.Dense
			jacobian.Sub(m.NewDiagDense(len(singleOutput), singleOutput), &softMaxMul)

			var gradient m.Dense
			gradient.Mul(&jacobian, m.NewDense(len(singleOutput), 1, singleDVals))
			dInputs.SetRow(i, gradient.RawMatrix().Data)
		}

		act.dInputs = *dInputs
	}

	return act
}

func softMaxLossCategoricalCrossEntropyActivation(target []float64) *activationFunc {
	act := &activationFunc{}

	act.Forward = func(inputs m.Dense) {
		act.inputs = inputs
		softMaxAct := softMaxActivation()

		softMaxAct.Forward(inputs)

		//l := categoricalCrossEntropyLoss(softMaxAct.output, target)
		//accu := accuracy(softMaxAct.output, target)

		//fmt.Println("loss:", meanLoss(l), "accuracy", accu)

		act.output = softMaxAct.output
	}

	act.Backward = func(dVals m.Dense) {
		r, c := dVals.Dims()

		dInputs := m.NewDense(r, c, nil)
		dInputs.CloneFrom(&dVals)

		for i := 0; i < r; i++ {
			classIndex := int(target[i])
			for j := 0; j < c; j++ {
				if j == classIndex {
					dInputs.Set(i, j, dInputs.At(i, j)-1)
				}
			}
		}

		var normalize m.Dense
		normalize.Apply(func(_, _ int, v float64) float64 {
            return v / float64(r)
        }, dInputs)

		act.dInputs = normalize
	}

	return act
}

func sigmoidActivation() *activationFunc {
    act := &activationFunc{}

    act.Forward = func(inputs m.Dense) {
        act.inputs = inputs
        act.output.Apply(func(i, j int, v float64) float64 {
            return 1 / (1 + math.Exp(-v))
        }, &act.inputs)
    }

    act.Backward = func(dVals m.Dense) {
        act.dInputs.Apply(func(i, j int, v float64) float64 {
            return dVals.At(i, j) * (1-v) * v
        }, &act.output)
    }

    return act
}

func linearActivation() *activationFunc {
    act := &activationFunc{}

    act.Forward = func(inputs m.Dense) {
        act.inputs = inputs
        act.output = inputs
    }

    act.Backward = func(dVals m.Dense) {
        act.dInputs = dVals
    }

    return act
}

/*var stepActivation activationFunc = activationFunc{
	forwardFunc: func(_, _ int, x float64) float64 {
		if x > 0 {
			return 1
		}
		return 0
	},
}

var sigmoidActivation activationFunc = activationFunc{
	forwardFunc: func(_, _ int, x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) },
}

var linearActivation activationFunc = activationFunc{
	forwardFunc: func(_, _ int, x float64) float64 { return x },
}*/
