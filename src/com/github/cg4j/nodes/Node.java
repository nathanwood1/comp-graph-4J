package com.github.cg4j.nodes;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;

/**
 * This interface specifies the requirements to be a node.
 * @since 1.0
 */
public interface Node {
    /**
     * Returns the shape of the output.
     * @return {@code int[]} shape.
     * @since 1.0
     */
    int[] getShape();

    /**
     * Returns a list of all children connected to this node.
     * @return {@code Node[]} of children.
     * @since 1.0
     */
    Node[] getChildren();

    /**
     * The method for calculating the output.
     * Do not use this method; use {@code Eval#evaluate} instead.
     * @see Eval
     * @param e The evaluator. This can be used to get the value of children.
     * @return The {@code Tensor} output.
     * @since 1.0
     */
    Tensor evaluate(Eval e);
}