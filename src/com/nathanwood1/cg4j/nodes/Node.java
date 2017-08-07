package com.nathanwood1.cg4j.nodes;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.nodes.io.VariableNode;
import com.nathanwood1.cg4j.optimizers.Optimizer;

import java.util.Arrays;
import java.util.HashMap;

//TODO: ADD JAVADOC FOR ALL NODES
public abstract class Node {
    private static int stringGlobalCounter = 0;

    public final String name;
    public final int[] shape;
    public final int length;

    protected Node[] children;

    public Node(int[] shape, String name, Node... children) {
        this.children = children;
        this.shape = shape;
        if (name == null) {
            this.name = getNodeClassName() + "_" + stringGlobalCounter++;
        } else {
            this.name = name;
        }
        int length = 1;
        for (int x : shape) {
            length *= x;
        }
        this.length = length;
    }

    public static boolean ShapeEndCompatible(int[] aShape, int aStart, int[] bShape, int bStart) {
        for (int i = 1; i <= Math.min(aShape.length - bStart, bShape.length - aStart); i++) {
            if (aShape[aShape.length - i - bStart] != bShape[bShape.length - i - aStart]) {
                return false;
            }
        }
        return true;
    }

    protected abstract String getNodeClassName();

    public Node[] getChildren() {
        return children;
    }

    public abstract Tensor evaluate(Eval e);

    public abstract void createGradients(HashMap<VariableNode, Node> deltas, Node parentDelta);

    @Override
    public String toString() {
        return "(" + name + ", " + Arrays.toString(children) + ")";
    }
}
