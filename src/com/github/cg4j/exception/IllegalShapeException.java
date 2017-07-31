package com.github.cg4j.exception;

/**
 * This exception represents a malformed shape.
 * @since 1.0
 * @author nathanwood1
 */
public class IllegalShapeException extends IllegalArgumentException {

    public IllegalShapeException(String str) {
        super(str);
    }
}
