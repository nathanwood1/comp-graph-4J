package com.nathanwood1.cg4j.nodes.math;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.exception.IllegalShapeException;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.VariableNode;
import com.nathanwood1.cg4j.optimizers.Optimizer;

import java.util.Arrays;
import java.util.HashMap;

public class SubtractionNode extends Node {
    public SubtractionNode(String name, Node a, Node b) {
        super(a.shape, name, a, b);
        if (!Node.ShapeEndCompatible(a.shape, 0, b.shape, 0)) {
            throw new IllegalShapeException(
                    "Ends of the shapes must be equal ("
                            + Arrays.toString(a.shape)
                            + ", "
                            + Arrays.toString(b.shape)
                            + ")"
            );
        }
    }

    public SubtractionNode(Node a, Node b) {
        super(a.shape, null, a, b);
        if (!Node.ShapeEndCompatible(a.shape, 0, b.shape, 0)) {
            throw new IllegalShapeException(
                    "Ends of the shapes must be equal ("
                            + Arrays.toString(a.shape)
                            + ", "
                            + Arrays.toString(b.shape)
                            + ")"
            );
        }
    }

    @Override
    protected String getNodeClassName() {
        return "SubtractionNode";
    }

    /**
     * Use {@code Eval#evaluate(Node)}
     *
     * @see Eval#evaluate(Node)
     */
    @Override
    public Tensor evaluate(Eval e) {
        Tensor aIn = e.evaluate(children[0]);
        Tensor bIn = e.evaluate(children[1]);
        Tensor out = new Tensor(new float[aIn.length], aIn.shape);
        for (int i = 0; i < out.length; i++) {
            out.setVal(i, aIn.getVal(i) - bIn.getVal(i));
        }
        return out;
    }

    @Override
    public void createGradients(HashMap<VariableNode, Node> deltas, Node parentDelta) {
        children[0].createGradients(deltas, parentDelta);
        children[1].createGradients(deltas, new NegationNode(name + "_Gradient", parentDelta));
    }
}
