package main

import (
	"log"
	"math"

	m "gonum.org/v1/gonum/mat"
)

func categoricalCrossEntropyLoss(inputs m.Dense, target []float64) m.Dense {
	inputs = clipOutputs(inputs)

	r, _ := inputs.Dims()

	predictedVals := m.NewDense(1, len(target), nil)
	for i := 0; i < r; i++ {
		row := inputs.RawRowView(i)
		for j, v := range row {
			if target[i] == float64(j) {
				predictedVals.Set(0, i, v)
				break
			}
		}
	}

	var losses m.Dense
	losses.Apply(func(_, _ int, v float64) float64 { return -math.Log(v) }, predictedVals)

	return losses
}

func backwardsLossCategoricalCrossEntropy(dVals m.Dense, target []float64) m.Dense {
	oneHotTarget := convertLabelToOneHot(target)

	_, samples := dVals.Dims()

	var dInputs m.Dense
	dInputs.Apply(func(i, j int, v float64) float64 { return -((oneHotTarget[i][j] / v) / float64(samples)) }, &dVals)
	return dInputs
}

func regularizationLoss(l layerDense) float64 {
	var regLoss float64

	if l.wRegularizerL1 > 0 {
		var wAbsMat m.Dense
		wAbsMat.Apply(func(i, j int, v float64) float64 { return math.Abs(v) }, &l.weights)
		regLoss += l.wRegularizerL1 * m.Sum(&wAbsMat)
	}

	if l.wRegularizerL2 > 0 {
		var wSqMat m.Dense
		wSqMat.Apply(func(i, j int, v float64) float64 { return v * v }, &l.weights)
		regLoss += l.wRegularizerL2 * m.Sum(&wSqMat)
	}

	if l.wRegularizerL1 > 0 {
		var bAbsSum float64
		for _, v := range l.biases {
			bAbsSum += math.Abs(v)
		}
		regLoss += l.wRegularizerL1 * bAbsSum
	}

	if l.wRegularizerL2 > 0 {
		var bSqSum float64
		for _, v := range l.biases {
			bSqSum += v * v
		}
		regLoss += l.wRegularizerL2 * bSqSum
	}

	return regLoss
}

func binaryCrossEntropyLoss(inputs m.Dense, target []float64) m.Dense {
    inputs = clipOutputs(inputs)

    r, _ := inputs.Dims()
    if len(target) != r {
        log.Fatal("target size mismatch with inputs")
    }

    var losses m.Dense
    losses.Apply(func(i, j int, v float64) float64 {
        return -(target[i] * math.Log(v) + (1 - target[i])*math.Log(1 - v))
    }, &inputs)

    return *m.DenseCopyOf(losses.T())
}

func backwardsLossBinaryCrossEntropy(dVals m.Dense, target []float64) m.Dense {
    dVals = clipOutputs(dVals)

    r, c := dVals.Dims()
    if len(target) != r {
        log.Fatal("target size mismatch with dvals")
    }

    var dInputs m.Dense
    dInputs.Apply(func(i, j int, v float64) float64 {
        return -((target[i] / v) - ((1-target[i])/(1-v))) / (float64(c*r))
    }, &dVals)

    return dInputs
}

func meanSquaredErrorLoss(inputs m.Dense, target []float64) m.Dense {
    var squaredDiff m.Dense
    squaredDiff.Apply(func(i, j int, v float64) float64 {
        return math.Pow(target[i] - v, 2)
    }, &inputs)

    return squaredDiff
}

func backwardsLossMeanSquaredError(dVals m.Dense, target []float64) m.Dense {
    r, c := dVals.Dims()
    if len(target) != r {
        log.Fatal("target size mismatch with dvals")
    }


    var dInputs m.Dense
    dInputs.Apply(func(i, j int, v float64) float64 {
        return -2 * (target[i] - v) / float64(c*r)
    }, &dVals)

    return dInputs
}

func meanAbsErrorLoss(inputs m.Dense, target []float64) m.Dense {
    var squaredDiff m.Dense
    squaredDiff.Apply(func(i, j int, v float64) float64 {
        return math.Abs(target[i] - v)
    }, &inputs)

    return squaredDiff
}

func backwardsLossMeanAbsError(dVals m.Dense, target []float64) m.Dense {
    r, c := dVals.Dims()
    if len(target) != r {
        log.Fatal("target size mismatch with dvals")
    }


    var dInputs m.Dense
    dInputs.Apply(func(i, j int, v float64) float64 {
        if (target[i] - v) > 0 {
            return 1 / float64(r * c)
        } else if (target[i] - v) < 0 {
            return -1 / float64(r * c)
        } else {
            return 0
        }
    }, &dVals)

    return dInputs
}
