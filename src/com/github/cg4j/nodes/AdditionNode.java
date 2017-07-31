package com.github.cg4j.nodes;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;
import com.github.cg4j.exception.IllegalShapeException;

import java.util.Arrays;

/**
 * A node for adding tensors.
 * @since 1.0
 * @author nathanwood1
 */
public class AdditionNode extends Node {
    private final Node childA;
    private final Node childB;
    private final boolean aDominant;

    private final int[] shape;

    /**
     * Creates an addition node from two children nodes.
     * @param childA
     * @param childB
     * @throws IllegalShapeException if the shapes are not comatable.
     * The end must be the same, for example, this is permitted:
     * <pre>{@code
     * a = [2, 4, 3]
     * b =    [4, 3]
     * }</pre>
     * But this is not:
     * <pre>{@code
     * a = [2, 4, 3]
     * b = [2, 4]
     * }</pre>
     * You are also allowed to have a single value, e.g.
     * <pre>{@code
     * a = [2, 4, 3]
     * b = [1]
     * }</pre>
     * @since 1.0
     */
    public AdditionNode(Node childA, Node childB) {
        aDominant = childA.getShape().length >= childB.getShape().length; // Check if a has more dimensions that b
        if (aDominant) {
            if (childB.getShape()[0] != 1) { // Allow a node with shape [1]
                for (int i = 1; i <= childB.getShape().length; i++) { // Check the last bits of the array are equal; see javadoc for more info
                    if (childA.getShape()[childA.getShape().length - i] != childB.getShape()[childB.getShape().length - i]) {
                        throw new IllegalShapeException(
                                "The end(s) of shape A must match shape B ("
                                        + Arrays.toString(childA.getShape())
                                        + ", "
                                        + Arrays.toString(childB.getShape())
                                        + ")"
                        );
                    }
                }
            }
            this.shape = childA.getShape();
            this.childA = childA;
            this.childB = childB;
        } else {
            if (childA.getShape()[0] != 1) {
                for (int i = 1; i <= childA.getShape().length; i++) {
                    if (childB.getShape()[childB.getShape().length - i] != childA.getShape()[childA.getShape().length - i]) {
                        throw new IllegalShapeException(
                                "The end(s) of shape B must match shape A ("
                                        + Arrays.toString(childA.getShape())
                                        + ", "
                                        + Arrays.toString(childB.getShape())
                                        + ")"
                        );
                    }
                }
            }
            this.shape = childA.getShape();
            this.childA = childA;
            this.childB = childB;
        }
    }

    /**
     * Returns the shape of the node.
     * @return {@code int[]} shape
     * @since 1.0
     */
    @Override
    public int[] getShape() {
        return shape;
    }

    /**
     * Returns the children of the node.
     * @return {@code Node[]} children
     * @since 1.0
     */
    @Override
    public Node[] getChildren() {
        return new Node[]{childA, childB};
    }

    /**
     * Calculates the value. Do not use this; use {@code Eval#evaluate} instead
     * @param e The evaluator. This can be used to get the value of children.
     * @return The value.
     */
    @Override
    public Tensor evaluate(Eval e) {
        Tensor a = e.evaluate(childA);
        Tensor b = e.evaluate(childB);

        if (aDominant) {
            Tensor out = new Tensor(new float[a.length], a.getShape());

            for (int i = 0; i < out.length; i++) {
                out.setVal(i, a.getVal(i) + b.getVal(i % b.length));
            }

            return out;
        } else {
            Tensor out = new Tensor(new float[b.length], b.getShape());

            for (int i = 0; i < out.length; i++) {
                out.setVal(i, a.getVal(i % a.length) + b.getVal(i));
            }

            return out;
        }
    }
}
