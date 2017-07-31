package com.github.cg4j.nodes;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;

/**
 * Node for feeding values into the graph.
 * @since 1.0
 * @author nathanwood1
 */
public class InputNode implements Node {
    private int[] shape;

    /**
     * Creates an {@code InputNode} with specified shape.
     * @param shape The shape to use.
     * @since 1.0
     */
    public InputNode(int[] shape) {
        this.shape = shape;
    }

    /**
     * Returns the shape.
     * @return {@code int[]} shape.
     * @since 1.0
     */
    @Override
    public int[] getShape() {
        return shape;
    }

    /**
     * Returns the children. Since this is an input node, there are no children.
     * @return {@code new Node[0];}
     * @since 1.0
     */
    @Override
    public Node[] getChildren() {
        return new Node[0];
    }

    /**
     * Returns the input data.
     * Do not use this; use {@code Eval#evaluate} instead.
     * @see Eval
     * @param e The evaluator. This can be used to get the value of children.
     * @return The input data.
     */
    @Override
    public Tensor evaluate(Eval e) {
        return e.evaluate(this);
    }
}
