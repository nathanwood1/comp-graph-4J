package com.nathanwood1.cg4j.optimizers;

import com.nathanwood1.cg4j.Eval;
import com.nathanwood1.cg4j.nodes.Node;
import com.nathanwood1.cg4j.nodes.io.VariableNode;

import java.util.HashMap;

public abstract class Optimizer {
    public HashMap<VariableNode, Node> deltas;

    public Optimizer() {
        deltas = new HashMap<>();
    }

    public abstract void minimize(Node node);

    public abstract void run(Eval eval);

    public void allowOptimize(VariableNode node) {
        deltas.putIfAbsent(node, null);
    }

    public void put(VariableNode node, Node nodeDelta) {
        if (deltas.containsKey(node)) {
            deltas.put(node, nodeDelta);
        }
    }
}
