package com.github.cg4j.nodes.io;

import com.github.cg4j.Eval;
import com.github.cg4j.Tensor;
import com.github.cg4j.nodes.Node;
import com.github.cg4j.optimizers.Optimizer;

public class InputNode extends Node {

    /**
     * Creates an {@code InputNode} from a shape.
     * @see Node
     * @param shape    The shape of the output.
     * @since 1.0
     */
    public InputNode(int[] shape) {
        super(shape, null);
    }

    /**
     * Creates an {@code InputNode} from a name and shape.
     * @see Node
     * @param shape    The shape of the output.
     * @param name     {@code String} name of node. Can be null.
     * @since 1.0
     */
    public InputNode(int[] shape, String name) {
        super(shape, name);
    }

    /**
     * Returns unique name for this node.
     * @return {@code "InputNode"}
     * @since 1.0
     */
    @Override
    public String nodeClassName() {
        return "InputNode";
    }

    /**
     * Use {@code Eval#evaluate(Node)}
     * @see Eval#evaluate(Node)
     */
    @Override
    public Tensor evaluate(Eval e) {
        return null;
    }

    @Override
    public void createGradients(Optimizer optimizer, Node parentDelta) {
    }
}
