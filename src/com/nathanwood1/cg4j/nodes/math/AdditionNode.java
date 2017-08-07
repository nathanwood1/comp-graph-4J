package com.nathanwood1.cg4j.nodes.math;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.exception.IllegalShapeException;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.VariableNode;
import com.nathanwood1.cg4j.optimizers.Optimizer;

import java.util.Arrays;
import java.util.HashMap;

public class AdditionNode extends Node {
    public AdditionNode(String name, Node... children) {
        super(children[0].shape, name, children);
        for (int i = 1; i < children.length; i++) {
            if (!Node.ShapeEndCompatible(children[0].shape, 0, children[i].shape, 0)) {
                throw new IllegalShapeException(
                        "Cannot add shapes ("
                                + Arrays.toString(children[0].shape)
                                + ", "
                                + Arrays.toString(children[i].shape)
                                + ")"
                );
            }
        }
    }

    public AdditionNode(Node... children) {
        super(children[0].shape, null, children);
        for (int i = 1; i < children.length; i++) {
            if (!Node.ShapeEndCompatible(children[0].shape, 0, children[i].shape, 0)) {
                throw new IllegalShapeException(
                        "Cannot add shapes ("
                                + Arrays.toString(children[0].shape)
                                + ", "
                                + Arrays.toString(children[i].shape)
                                + ")"
                );
            }
        }
    }

    @Override
    protected String getNodeClassName() {
        return "Addition";
    }

    /**
     * Use {@code Eval#evaluate(Node)}
     *
     * @see Eval#evaluate(Node)
     */
    @Override
    public Tensor evaluate(Eval e) {
        if (children.length == 1) {
            return e.evaluate(children[0]);
        }
        Tensor out = null;
        boolean init = false;
        for (Node child : children) {
            Tensor in = e.evaluate(child);
            if (!init) {
                out = new Tensor(new float[in.length], in.shape);
                for (int i = 0; i < out.length; i++) {
                    out.setVal(i, in.getVal(i % in.length));
                }
            } else {
                for (int i = 0; i < out.length; i++) {
                    out.setVal(i, out.getVal(i) + in.getVal(i % in.length));
                }
            }
            init = true;
        }
        return out;
    }

    /**
     * Creates the gradients.
     * <pre>
     * {@literal
     * f(x, y) = x + y
     *
     * ∂f/∂x = ∂/∂x[x + y]
     *       = ∂/∂x[x] + ∂/∂x[y]
     *       = 1 + 0
     *       = 1
     *
     * 1 * parentDelta = parentDelta
     * }
     * </pre>
     * The output is {@code parentDelta}, so we forward the output to all our children.
     * @param deltas The deltas of all variables.
     * @param parentDelta Last node's delta.
     */
    @Override
    public void createGradients(HashMap<VariableNode, Node> deltas, Node parentDelta) {
        for (Node child : children) {
            child.createGradients(deltas, parentDelta);
        }
    }
}
