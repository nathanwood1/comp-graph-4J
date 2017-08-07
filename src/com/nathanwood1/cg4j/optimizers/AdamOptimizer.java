package com.nathanwood1.cg4j.optimizers;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.Tensor;
import com.nathanwood1.cg4j.exception.NoVariableNodeToMinimizeException;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.VariableNode;

import java.util.HashMap;
import java.util.LinkedList;

public class AdamOptimizer extends Optimizer {
    public LinkedList<VariableNode> toTweak;

    public Node toChange = null;
    public boolean minimizing;

    public float learningRate = 0.001f;
    public float beta1 = 0.9f, beta2 = 0.999f;
    public float epsilon = 10E-8f;

    private float beta1PowerT = Float.NaN;
    private float beta2PowerT;

    private HashMap<VariableNode, Tensor> momentM;
    private HashMap<VariableNode, Tensor> momentV;

    public AdamOptimizer() {
        momentM = new HashMap<>();
        momentV = new HashMap<>();

        toTweak = new LinkedList<>();
    }

    public AdamOptimizer tweak(VariableNode toTweak) {
        this.toTweak.add(toTweak);
        return this;
    }

    @Override
    public void minimize(Node toMinimize, HashMap<VariableNode, Node> deltas) {
        this.toChange = toMinimize;
        minimizing = true;
        this.deltas = deltas;

        beta1PowerT = beta1;
        beta2PowerT = beta2;

        for (VariableNode var : toTweak) {
            momentM.put(var, new Tensor(new float[var.val.length], var.val.shape));
            momentV.put(var, new Tensor(new float[var.val.length], var.val.shape));
        }
    }

    @Override
    public void maximize(Node node, HashMap<VariableNode, Node> deltas) {
        this.toChange = toChange;
        minimizing = false;
        this.deltas = deltas;

        beta1PowerT = beta1;
        beta2PowerT = beta2;

        for (VariableNode var : toTweak) {
            momentM.put(var, new Tensor(new float[var.val.length], var.val.shape));
            momentV.put(var, new Tensor(new float[var.val.length], var.val.shape));
        }
    }

    @Override
    public void run(Eval eval) {
        if (toChange == null) {
            throw new NoVariableNodeToMinimizeException("No nodes where told to minimize!");
        }
        if (minimizing) {
            for (VariableNode var : toTweak) {
                Tensor theta = var.val;

                Node deltaNode = deltas.get(var);
                Tensor delta = eval.evaluate(deltaNode);

                Tensor mLast = momentM.get(var);
                Tensor vLast = momentV.get(var);

                for (int i = 0; i < var.val.length; i++) {
                    float m = beta1 * mLast.getVal(i) + (1 - beta1) * delta.getVal(i);
                    float v = beta2 * vLast.getVal(i) + (1 - beta2) * delta.getVal(i) * delta.getVal(i);

                    float alphaT = (learningRate * (float) Math.sqrt(1 - beta2PowerT)) / (1 - beta1PowerT);
                    theta.setVal(i, theta.getVal(i) - (alphaT * m) / ((float) Math.sqrt(v) + epsilon));
                }
            }
        } else {
            for (VariableNode var : toTweak) {
                Tensor theta = var.val;

                Node deltaNode = deltas.get(var);
                Tensor delta = eval.evaluate(deltaNode);

                Tensor mLast = momentM.get(var);
                Tensor vLast = momentV.get(var);

                for (int i = 0; i < var.val.length; i++) {
                    float m = beta1 * mLast.getVal(i) + (1 - beta1) * delta.getVal(i);
                    float v = beta2 * vLast.getVal(i) + (1 - beta2) * delta.getVal(i) * delta.getVal(i);

                    float alphaT = (learningRate * (float) Math.sqrt(1 - beta2PowerT)) / (1 - beta1PowerT);
                    theta.setVal(i, theta.getVal(i) + (alphaT * m) / ((float) Math.sqrt(v) + epsilon));
                }
            }
        }
        beta1PowerT *= beta1PowerT;
        beta2PowerT *= beta2PowerT;
    }
}
