package main

import (
	"fmt"

	d "github.com/janscheres/nn/getting_data"
	//m "gonum.org/v1/gonum/mat"
)

func main() {
    //
    //
    // BINARY
    //
    //
    dense1 := newLayerDense(2, 64, 0, 5e-4, 0, 5e-4)

	act1 := reluActivation()

	//dense1.Forward(d.X)

	//dense1.output = activate(dense1.output, reluActivation)

	dense2 := newLayerDense(64, 1, 0, 0, 0, 0)
	act2 := sigmoidActivation()

	optimizer := newOptimizerAdam(0.001, 5e-7, 1e-7, 0.9, 0.999)

    for i := 0; i < 10001; i++ {
		dense1.Forward(*d.RawXTwo)
		act1.Forward(dense1.output)
		dense2.Forward(act1.output)
		act2.Forward(dense2.output)

		if i%100 == 0 {
			regLoss := regularizationLoss(dense1) + regularizationLoss(dense2)
            loss := binaryCrossEntropyLoss(act2.output, d.TwoY)

			fmt.Println(i, "accu", binaryAccuracy(act2.output, d.TwoY), "loss", meanLoss(loss)+regLoss, "normalLoss", meanLoss(loss))
		}

        din := backwardsLossBinaryCrossEntropy(act2.output, d.TwoY)
		act2.Backward(din)
		dense2.Backward(act2.dInputs)
		act1.Backward(dense2.dInputs)
		dense1.Backward(act1.dInputs)

		optimizer.preUpdateParams()
		optimizer.updateParams(&dense1)
		optimizer.updateParams(&dense2)
		optimizer.postUpdateParams()
	}

    fmt.Println("testing:")
	dense1.Forward(*d.RawXTwoTest)
	act1.Forward(dense1.output)
	dense2.Forward(act1.output)
	act2.Forward(dense2.output)

	regLoss := regularizationLoss(dense1) + regularizationLoss(dense2)

	fmt.Println("accu", binaryAccuracy(act2.output, d.TwoY), "loss", meanLoss(binaryCrossEntropyLoss(act2.output, d.TwoY))+regLoss)

    //
    //
    // CLASSIFICATION
    //
    //
	/*dense1 := newLayerDense(2, 64, 0, 5e-4, 0, 5e-4)

	act1 := reluActivation()

	dropOut1 := newDropOutLater(0.2)

	//dense1.Forward(d.X)

	//dense1.output = activate(dense1.output, reluActivation)

	dense2 := newLayerDense(64, 3, 0, 0, 0, 0)
	act2 := softMaxLossCategoricalCrossEntropyActivation(d.Y)

	//optimizer := newOptimizerSGD(1, 0.001, 0.9)
	//optimizer := newOptimizerAdaGrad(1, 1e-4, 1e-7)
	//optimizer := newOptimizerRMSProp(0.02, 1e-5, 1e-7, 0.999)
	optimizer := newOptimizerAdam(0.05, 5e-5, 1e-7, 0.9, 0.999)

	for i := 0; i < 10001; i++ {
		dense1.Forward(*d.X)
		act1.Forward(dense1.output)
		dropOut1.Forward(act1.output)
		dense2.Forward(dropOut1.output)
		act2.Forward(dense2.output)

		if i%100 == 0 {
			regLoss := regularizationLoss(dense1) + regularizationLoss(dense2)
			loss := categoricalCrossEntropyLoss(act2.output, d.Y)

			fmt.Println(i, "accu", accuracy(act2.output, d.Y), "loss", meanLoss(loss)+regLoss, "normalLoss", meanLoss(loss))
		}

		act2.Backward(act2.output)
		dense2.Backward(act2.dInputs)
		dropOut1.Backward(dense2.dInputs)
		act1.Backward(dropOut1.dInputs)
		dense1.Backward(act1.dInputs)

		optimizer.preUpdateParams()
		optimizer.updateParams(&dense1)
		optimizer.updateParams(&dense2)
		optimizer.postUpdateParams()
	}

	fmt.Println("testing:")
	// validate data with test set
	// difference of over 10% in training loss is a sign of serious overfitting
	// similar performace on both datasets means that model generalized instead of overfitting the training data
	// learning rate too high or too many epochs or model too big
	// if model too big, try find smallest possible model that still learns
	// fewer neurons means less chance of memorization, meaning easier to generalize
	// to find best generalization hyperparams, dont check with the test data as this will be manually biasing it, to overfitting
	// for said tuning, use the validation dataset
	// test data is only to be used as unseen data
	// search for hyperparams using validation an test at the end using the test to see ifit overfitted to the validation
	dense1.Forward(*d.XTest)
	act1.Forward(dense1.output)
	dense2.Forward(act1.output)
	act2.Forward(dense2.output)

	regLoss := regularizationLoss(dense1) + regularizationLoss(dense2)

	fmt.Println("accu", accuracy(act2.output, d.Y), "loss", meanLoss(categoricalCrossEntropyLoss(act2.output, d.Y))+regLoss)
    */

    //
    // OLD
    //
	//dense2.Forward(&dense1.output)

	//dense2.output = softMaxActivation(dense2.output)
	//printMat(&dense2.output)

	//losses := categoricalCrossEntropyLossOneHot(dense2.output, d.Y)
	//acc := accuracyOneHot(dense2.output, d.Y)
	//fmt.Println(acc)

	//fmt.Println(meanLoss(losses))

	/*fmt.Println(dense1.dWeights)
		fmt.Println(dense1.dBiases)
	    fmt.Println("/////////////////////////////////////////")
		fmt.Println(dense2.dWeights)
		fmt.Println(dense2.dBiases)*/
}

/*rawSoftMaxOutputs := []float64{
      0.7, 0.1, 0.2,
      0.1, 0.5, 0.4,
      0.02, 0.9, 0.08,
  }
  softMaxOutputs := m.NewDense(3, 3, rawSoftMaxOutputs)

  targets := []float64{0, 1, 1}

  softMaxLoss := backwardsSoftMaxLossCategoricalCrossEntropy(*softMaxOutputs, targets)
  printMat(&softMaxLoss)

  seperate := backwardSoftMaxActivation(*softMaxOutputs, backwardsLossCategoricalCrossEntropy(*softMaxOutputs, targets))
  printMat(seperate)*/

/*input := m.NewDense(2, 3, []float64{
      1.0, 2.0, 3.0,
      1.0, 2.0, 3.0,
  })

  // Forward pass
  output := softMaxActivation(*input)
  fmt.Printf("Softmax Output:\n%v\n\n", m.Formatted(&output))

  // Backward pass
  dvalues := m.NewDense(2, 3, []float64{
      0.1, 0.2, 0.7,
      0.2, 0.3, 0.5,
  })

  dinputs := backwardSoftMaxActivation(output, *dvalues)
  fmt.Printf("Softmax Backward:\n%v\n", m.Formatted(dinputs))*/

//rawSoftMaxOutputs := []float64{0.7, 0.1, 0.2}
/*softMaxOutputs := m.NewDense(1, 3, rawSoftMaxOutputs)
  diagSoftMaxOutputs := m.NewDiagDense(3, rawSoftMaxOutputs)
  fmt.Println("")

  var softMaxMul m.Dense
  softMaxMul.Mul(softMaxOutputs.T(), softMaxOutputs)
  printMat(&softMaxMul)

  printMat(diagSoftMaxOutputs)

  var jacobian m.Dense
  jacobian.Sub(diagSoftMaxOutputs, &softMaxMul)
  printMat(&jacobian)*/

//fmt.Println("hi")
//printMat(backwardsSoftMaxActivation(rawSoftMaxOutputs, *m.NewDense(3, 3, nil)))

/*rawSoftMaxOutputs := []float64{
        0.7, 0.1, 0.2,
        0.5, 0.1, 0.4,
        0.02, 0.9, 0.08,
    }
    softMaxOutputs := m.NewDense(3, 3, rawSoftMaxOutputs)

    targetOutput := [][]float64{
        {1, 0, 0},
        {0, 1, 0},
        {0, 1, 0},
    }

    acc := accuracy(*softMaxOutputs, targetOutput)
    fmt.Println(acc)

	losses := categoricalCrossEntropyLoss(*softMaxOutputs, targetOutput)

    fmt.Println(meanLoss(losses))*/

//dVals := m.NewDense(3, 3, []float64{1, 1, 1, 2, 2, 2, 3, 3, 3,})

/*rawInputs := []float64{
      1, 2, 3, 2.5,
      2, 5, -1, 2,
      -1.5, 2.7, 3.3, -0.8,
  }
  inputs := m.NewDense(3, 4, rawInputs)

  rawWeights := []float64{
      0.2, 0.8, -0.5, 1,
      0.5, -0.91, 0.26, -0.5,
      -0.26, -0.27, 0.17, 0.87,
  }
  weights := m.DenseCopyOf(m.NewDense(3, 4, rawWeights).T())

  rawBiases := []float64{2, 3, 0.5}
  biases := m.NewDense(1, 3, rawBiases)

  var layerOutputs m.Dense
  layerOutputs.Mul(inputs, weights)
  layerOutputs = addBiases(layerOutputs, rawBiases)

  reluOutputs := activate(layerOutputs, reluActivation)

  var dRelu m.Dense
  dRelu.Apply(func(i, j int, v float64) float64 { if layerOutputs.At(i, j) <= 0 {return 0} else {return v}}, &reluOutputs)

  var dInputs m.Dense
  dInputs.Mul(&dRelu, weights.T())

  var dWeights m.Dense
  dWeights.Mul(inputs.T(), &dRelu)


  _, c := dRelu.Dims()

  rawDBiases := make([]float64, c)
  for i := 0; i < c; i++ {
      row := m.DenseCopyOf(dRelu.T()).RawRowView(i)
      for _, v := range row {
          rawDBiases[i] += v
      }
  }
  dBiases := m.NewDense(1, 3, rawDBiases)

  weights.Apply(func(i, j int, v float64) float64 {return v + -0.001*dWeights.At(i, j)}, weights)
  biases.Apply(func(i, j int, v float64) float64 { return v + -0.001*dBiases.At(i, j)}, biases)

  printMat(weights)
  fmt.Println("//////////////////////")
  printMat(biases)*/
/*var dWeights m.Dense
  dWeights.Mul(m.DenseCopyOf(inputs.T()), dVals)
  printMat(&dWeights)

  _, c := dVals.Dims()

  rowSums := make([]float64, c)
  for i := 0; i < c; i++ {
      row := m.DenseCopyOf(dVals.T()).RawRowView(i)
      for _, v := range row {
          rowSums[i] += v
      }
  }

  fmt.Println(rowSums)

  //

      tWeights := m.DenseCopyOf(weights.T())

  var dInputs m.Dense
  dInputs.Mul(dVals, tWeights)
  printMat(&dInputs)

  //


  rawZ := []float64{
      1, 2, -3, -4,
      2, -7, -1, 3,
      -1, 2, 5, -1,
  }
  z := m.NewDense(3, 4, rawZ)

  var dRelu m.Dense

  dRelu.Apply(func(i, j int, v float64) float64 { if z.At(i, j) <= 0 {return 0} else {return v}}, dVals)
  printMat(&dRelu)*/

/*r, _ := tWeights.Dims()

  rowSums := make([]float64, r)
  dXs := make([]float64, r)
  for i := 0; i < r; i++ {
      row := tWeights.RawRowView(i)
      for _, v := range row {
          rowSums[i] += v
      }
      dXs[i] = rowSums[i] * dVals[0]
  }
  fmt.Println(rowSums)*/

/*rawInputs := []float64{
      1.0, 2.0, 3.0, 2.5,
      2.0, 5.0, -1.0, 2.0,
      -1.5, 2.7, 3.3, -0.8,
  }
  //layer one
  rawWeights1 := []float64{
      0.2, 0.8, -0.5, 1.0,
      0.5, -0.91, 0.26, -0.5,
      -0.26, -0.27, 0.17, 0.87,
  }
  biases1 := []float64{2.0, 3.0, 0.5}
  //layer two
  rawWeights2 := []float64{
      0.1, -0.14, 0.5,
      -0.5, 0.12, -0.33,
      -0.44, 0.73, -0.13,
  }
  biases2 := []float64{-1.0, 2.0, -0.5}

  inputs := m.NewDense(3, 4, rawInputs)
  weights1 := m.NewDense(3, 4, rawWeights1)

  tWeights1 := weights1.T()

  var dotProd1 m.Dense
  dotProd1.Mul(inputs, tWeights1)
  //printMat(&dotProd1)

  result1 := addBiases(dotProd1, biases1)

  weights2 := m.NewDense(3, 3, rawWeights2)
  tWeights2 := weights2.T()

  var dotProd2 m.Dense
  dotProd2.Mul(&result1, tWeights2)

  result2 := addBiases(dotProd2, biases2)
  printMat(&result2)*/
