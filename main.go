package main

/*import (
    "fmt"

    t "gorgonia.org/tensor"
)

func addBiases(inputs t.Tensor, biases []float32) (t.Tensor, error) {
    backing := make([]float32, inputs.Shape()[0]*inputs.Shape()[1])
    for i := range backing {
        backing[i] = biases[i%len(biases)]
    }

    return t.Add(inputs, t.New(t.WithShape(inputs.Shape()...), t.WithBacking(backing)))
}

func main() {
    rawInputs := []float32{
        1.0, 2.0, 3.0, 2.5,
        2.0, 5.0, -1.0, 2.0,
        -1.5, 2.7, 3.3, -0.8,
    }
    inputs := t.New(t.WithShape(3, 4), t.WithBacking(rawInputs))

    rawWeights := []float32{
        0.2, 0.8, -0.5, 1.0,
        0.5, -0.91, 0.26, -0.5,
        -0.26, -0.27, 0.17, 0.87,
    }
    weights := t.New(t.WithShape(3, 4), t.WithBacking(rawWeights))
    weights.T()

    bias := []float32{2.0, 3.0, 0.5}

    dotProd, err := t.Dot(inputs, weights)
    if err != nil {
        fmt.Println(err)
    }
    fmt.Println(dotProd)

    output, _ := addBiases(dotProd, bias)
    if err != nil {
        fmt.Println(err)
    }

    fmt.Println(output)
}*/
