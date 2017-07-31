package com.github.cg4j;

import com.github.cg4j.exception.IllegalShapeException;
import com.github.cg4j.exception.NoInputSpecifiedException;
import com.github.cg4j.nodes.InputNode;
import com.github.cg4j.nodes.Node;

import java.util.Arrays;
import java.util.HashMap;

/**
 * This class will calculate the graph.
 * @since 1.0
 * @author nathanwood1
 */
public class Eval {
    private final HashMap<Node, Tensor> tensorData;

    /**
     * Creates an {@code Eval}
     * @since 1.0
     */
    public Eval() {
        tensorData = new HashMap<>();
    }

    /**
     * Adds an input to the graph.
     * @param node An {@code InputNode} representing where in the graph the input should be added.
     * @param input The input to go into the graph.
     * @return This eval class. This allows {@code new Eval().addInput(...).addInput...}
     * @throws IllegalShapeException if the data's shape doesn't match the input node's.
     * @since 1.0
     */
    public Eval addInput(InputNode node, Tensor input) {
        int[] shape1 = node.getShape();
        int[] shape2 = input.getShape();

        if (shape1.length != shape2.length) {
            throw new IllegalShapeException(
                    "Input data doesn't have the same dimensionality as the input node ("
                            + Arrays.toString(shape1)
                            + ".length != "
                            + Arrays.toString(shape2)
                            + ".length)"
            );
        }
        for (int i = 0; i < shape1.length; i++) {
            if (shape1[i] != -1) {
                if (shape1[i] != shape2[i]) {
                    throw new IllegalShapeException(
                            "Input data doesn't have the same shape as input node ("
                                    + Arrays.toString(shape1)
                                    + " != "
                                    + Arrays.toString(shape2)
                                    + ")"
                    );
                }
            }
        }
        tensorData.put(node, input);
        return this;
    }

    /**
     * Calculate the value of a {@code Node}.
     * Use this instead of {@code Node#evaluate}.
     * @param node The node to evaluate.
     * @return The value.
     * @throws NoInputSpecifiedException if no input was given for this node.
     * @since 1.0
     */
    public Tensor evaluate(Node node) {
        if (tensorData.containsKey(node)) {
            return tensorData.get(node);
        } else {
            if (node instanceof InputNode) {
                throw new NoInputSpecifiedException(
                        "No input was specified for "
                        + node
                );
            }
            Tensor out = node.evaluate(this);
            tensorData.put(node, out);
            return out;
        }
    }
}
