package com.github.cg4j.nodes.io;

import com.github.cg4j.nodes.Node;
import com.github.cg4j.nodes.math.AdditionNode;

public class VariableDeltaNode extends AdditionNode {
    public VariableDeltaNode(Node... children) {
        super(children);
    }

    @Override
    public String getNodeClassName() {
        return "VariableDeltaNode";
    }

}
