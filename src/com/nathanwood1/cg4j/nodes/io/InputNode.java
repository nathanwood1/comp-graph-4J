package com.nathanwood1.cg4j.nodes.io;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.optimizers.Optimizer;

import java.util.HashMap;

/**
 * An input node feeds data into the graph.
 *
 * @author nathanwood1
 * @see Node
 * @since 1.0
 */
public class InputNode extends Node {

    /**
     * Creates an {@code InputNode} from a shape.
     *
     * @param shape The shape of the output.
     * @see Node
     * @since 1.0
     */
    public InputNode(int[] shape) {
        super(shape, null);
    }

    /**
     * Creates an {@code InputNode} from a name and shape.
     *
     * @param shape The shape of the output.
     * @param name  {@code String} name of node. Can be null.
     * @see Node
     * @since 1.0
     */
    public InputNode(int[] shape, String name) {
        super(shape, name);
    }

    @Override
    protected String getNodeClassName() {
        return "InputNode";
    }

    /**
     * Use {@code Eval#evaluate(Node)}
     *
     * @see Eval#evaluate(Node)
     */
    @Override
    public Tensor evaluate(Eval e) {
        return null;
    }

    @Override
    public void createGradients(HashMap<VariableNode, Node> deltas, Node parentDelta) {
    }
}
