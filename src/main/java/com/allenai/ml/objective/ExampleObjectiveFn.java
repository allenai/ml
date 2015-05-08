package com.allenai.ml.objective;

import com.allenai.ml.linalg.Vector;

/**
 * Interface for updating a single example. Meant to be used with a `BatchObjectiveFn` to setup the
 * optimization function for a machine learning problem.
 */
public interface ExampleObjectiveFn<T> {
    /**
     *
     * @param example The example to operate on
     * @param inParams read-only version of the current paramters
     * @param outGrad Meant to be mutated by the function to reflect the gradient updates from this example
     * @return The objective function value for this example.
     */
    double evaluate(T example, Vector inParams, Vector outGrad);

}
