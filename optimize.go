package main

import (
	"math"

	m "gonum.org/v1/gonum/mat"
)

// stochastic gradient descent
type optimizerSGD struct {
	learningRate        float64
	currentLearningRate float64
	decay               float64
	iterations          float64
	momentum            float64
}

func newOptimizerSGD(learningRate, decay, momentum float64) optimizerSGD {
	// e.g.: 1.0, 0.0 (down to 0.1), 0.0
	return optimizerSGD{learningRate: learningRate, currentLearningRate: learningRate, decay: decay, iterations: 0, momentum: momentum}
}

func (o *optimizerSGD) preUpdateParams() {
	if o.decay != 0 {
		o.currentLearningRate = o.learningRate * (1 / (1 + (o.decay * o.iterations)))
	}
}

func (o *optimizerSGD) updateParams(layer *layerDense) {
	if o.momentum != 0 {
		// if doesnt contain momentum arrays, create and fill with zeros
		if layer.weightMomentums.IsEmpty() || len(layer.biasMomentums) == 0 {
			r, c := layer.weights.Dims()
			layer.weightMomentums = *m.NewDense(r, c, nil)
			layer.biasMomentums = make([]float64, len(layer.biases))
		}

		layer.weights.Apply(func(i, j int, v float64) float64 {
			weightUpdate := o.momentum*layer.weightMomentums.At(i, j) - o.currentLearningRate*layer.dWeights.At(i, j)
			layer.weightMomentums.Set(i, j, weightUpdate)
			return v + weightUpdate
		}, &layer.weights)

		for k, v := range layer.biases {
			biasUpdate := o.momentum*layer.biasMomentums[k] - o.currentLearningRate*layer.dBiases.At(0, k)
			layer.biasMomentums[k] = biasUpdate
			layer.biases[k] = v + biasUpdate
		}
	} else {
		layer.weights.Apply(func(i, j int, v float64) float64 {
			return v - (o.currentLearningRate * layer.dWeights.At(i, j))
		}, &layer.weights)

		for k, v := range layer.biases {
			layer.biases[k] = v - (o.currentLearningRate * layer.dBiases.At(0, k))
		}
	}
}

func (o *optimizerSGD) postUpdateParams() {
	o.iterations += 1
}

// adaptive gradients
type optimizerAdaGrad struct {
	learningRate        float64
	currentLearningRate float64
	decay               float64
	iterations          float64
	epsilon             float64
}

func newOptimizerAdaGrad(learningRate, decay, epsilon float64) optimizerAdaGrad {
	// e.g.: 1.0, 0.0, 1e-7
	return optimizerAdaGrad{learningRate: learningRate, currentLearningRate: learningRate, decay: decay, iterations: 0, epsilon: epsilon}
}

func (o *optimizerAdaGrad) preUpdateParams() {
	if o.decay != 0 {
		o.currentLearningRate = o.learningRate * (1 / (1 + (o.decay * o.iterations)))
	}
}

func (o *optimizerAdaGrad) updateParams(layer *layerDense) {
	// if doesnt contain cache arrays, create and fill with zeros
	if layer.weightCache.IsEmpty() || len(layer.biasCache) == 0 {
		r, c := layer.weights.Dims()
		layer.weightCache = *m.NewDense(r, c, nil)
		layer.biasCache = make([]float64, len(layer.biases))
	}

	layer.weights.Apply(func(i, j int, v float64) float64 {
		dWeight := layer.dWeights.At(i, j)
		cache := layer.weightCache.At(i, j) + (dWeight * dWeight)
		layer.weightCache.Set(i, j, cache)

		return v - ((o.currentLearningRate * dWeight) / (math.Sqrt(cache) + o.epsilon))
	}, &layer.weights)

	for k, v := range layer.biases {
		dBias := layer.dBiases.At(0, k)
		cache := layer.biasCache[k] + (dBias * dBias)
		layer.biasCache[k] = cache

		layer.biases[k] = v - ((o.currentLearningRate * dBias) / (math.Sqrt(cache) + o.epsilon))
	}
}

func (o *optimizerAdaGrad) postUpdateParams() {
	o.iterations += 1
}

// root mean square propogation
type optimizerRMSProp struct {
	learningRate        float64
	currentLearningRate float64
	decay               float64
	iterations          float64
	epsilon             float64
	rho                 float64
}

func newOptimizerRMSProp(learningRate, decay, epsilon, rho float64) optimizerRMSProp {
	// e.g.: 0.001, 0.0, 1e-7, 0.9
	return optimizerRMSProp{learningRate: learningRate, currentLearningRate: learningRate, decay: decay, iterations: 0, epsilon: epsilon, rho: rho}
}

func (o *optimizerRMSProp) preUpdateParams() {
	if o.decay != 0 {
		o.currentLearningRate = o.learningRate * (1 / (1 + (o.decay * o.iterations)))
	}
}

func (o *optimizerRMSProp) updateParams(layer *layerDense) {
	// if doesnt contain cache arrays, create and fill with zeros
	if layer.weightCache.IsEmpty() || len(layer.biasCache) == 0 {
		r, c := layer.weights.Dims()
		layer.weightCache = *m.NewDense(r, c, nil)
		layer.biasCache = make([]float64, len(layer.biases))
	}

	layer.weights.Apply(func(i, j int, v float64) float64 {
		dWeight := layer.dWeights.At(i, j)
		cache := (o.rho * layer.weightCache.At(i, j)) + ((1 - o.rho) * (dWeight * dWeight))
		layer.weightCache.Set(i, j, cache)

		return v - ((o.currentLearningRate * dWeight) / (math.Sqrt(cache) + o.epsilon))
	}, &layer.weights)

	for k, v := range layer.biases {
		dBias := layer.dBiases.At(0, k)
		cache := (o.rho * layer.biasCache[k]) + ((1 - o.rho) * (dBias * dBias))
		layer.biasCache[k] = cache

		layer.biases[k] = v - ((o.currentLearningRate * dBias) / (math.Sqrt(cache) + o.epsilon))
	}
}

func (o *optimizerRMSProp) postUpdateParams() {
	o.iterations += 1
}

// adaptive momentum
type optimizerAdam struct {
	learningRate        float64
	currentLearningRate float64
	decay               float64
	iterations          float64
	epsilon             float64
	beta1               float64
	beta2               float64
}

func newOptimizerAdam(learningRate, decay, epsilon, beta1, beta2 float64) optimizerAdam {
	// e.g.: 0.001, 0.0 (down to 1e-4), 1e-7, 0.9, 0.999
	return optimizerAdam{learningRate: learningRate, currentLearningRate: learningRate, decay: decay, iterations: 0, epsilon: epsilon, beta1: beta1, beta2: beta2}
}

func (o *optimizerAdam) preUpdateParams() {
	if o.decay != 0 {
		o.currentLearningRate = o.learningRate * (1 / (1 + (o.decay * o.iterations)))
	}
}

func (o *optimizerAdam) updateParams(layer *layerDense) {
	// if doesnt contain cache arrays, create and fill with zeros
	if layer.weightMomentums.IsEmpty() || layer.weightCache.IsEmpty() || len(layer.biasMomentums) == 0 || len(layer.biasCache) == 0 {
		r, c := layer.weights.Dims()
		layer.weightMomentums = *m.NewDense(r, c, nil)
		layer.weightCache = *m.NewDense(r, c, nil)
		layer.biasMomentums = make([]float64, len(layer.biases))
		layer.biasCache = make([]float64, len(layer.biases))
	}

	beta1Power := math.Pow(o.beta1, o.iterations+1)
	beta2Power := math.Pow(o.beta2, o.iterations+1)

	layer.weights.Apply(func(i, j int, v float64) float64 {
		dWeight := layer.dWeights.At(i, j)

		wMom := (o.beta1 * layer.weightMomentums.At(i, j)) + ((1 - o.beta1) * dWeight)
		layer.weightMomentums.Set(i, j, wMom)

		wMomCorr := wMom / (1 - beta1Power)

		wCache := (o.beta2 * layer.weightCache.At(i, j)) + ((1 - o.beta2) * (dWeight * dWeight))
		layer.weightCache.Set(i, j, wCache)

		wCacheCorr := wCache / (1 - beta2Power)

		return v - ((o.currentLearningRate * wMomCorr) / (math.Sqrt(wCacheCorr) + o.epsilon))
	}, &layer.weights)

	for k, v := range layer.biases {
		dBias := layer.dBiases.At(0, k)

		bMom := (o.beta1 * layer.biasMomentums[k]) + ((1 - o.beta1) * dBias)
		layer.biasMomentums[k] = bMom

		bMomCorr := bMom / (1 - beta1Power)

		bCache := (o.beta2 * layer.biasCache[k]) + ((1 - o.beta2) * (dBias * dBias))
		layer.biasCache[k] = bCache

		bCacheCorr := bCache / (1 - beta2Power)

		layer.biases[k] = v - ((o.currentLearningRate * bMomCorr) / (math.Sqrt(bCacheCorr) + o.epsilon))
	}
}

func (o *optimizerAdam) postUpdateParams() {
	o.iterations += 1
}
