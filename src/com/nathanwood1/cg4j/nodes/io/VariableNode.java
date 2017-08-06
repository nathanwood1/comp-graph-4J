package com.nathanwood1.cg4j.nodes.io;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.math.AdditionNode;
import com.nathanwood1.cg4j.optimizers.Optimizer;

public class VariableNode extends Node {
    public final Tensor val;
    private boolean gradientCreated = false;

    public VariableNode(Tensor val) {
        super(val.shape, null);
        this.val = val;
    }

    public VariableNode(Tensor val, String name) {
        super(val.shape, name);
        this.val = val;
    }

    @Override
    public String getNodeClassName() {
        return "VariableNode";
    }

    @Override
    public Tensor evaluate(Eval e) {
        return val;
    }

    @Override
    public void createGradients(Optimizer optimizer, Node parentDelta) {
        if (gradientCreated) {
            ((VariableDeltaNode) optimizer.deltas.get(this)).addChild(parentDelta);
        } else {
            if (!optimizer.deltas.containsKey(this)) {
                return;
            }
            optimizer.deltas.put(this, new VariableDeltaNode(parentDelta));
            gradientCreated = true;
        }
    }

    private class VariableDeltaNode extends AdditionNode {
        public VariableDeltaNode(Node... children) {
            super(children);
        }

        public void addChild(Node child) {
            Node[] childrenU = new Node[children.length + 1];
            System.arraycopy(children, 0, childrenU, 0, children.length);
            childrenU[children.length] = child;
            this.children = childrenU;
        }

        @Override
        public String getNodeClassName() {
            return "VariableDeltaNode";
        }

    }

}
