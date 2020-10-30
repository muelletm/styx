package com.stupid.styx;

public class Timer {
    private final long time_;

    public Timer() {
        time_ = System.currentTimeMillis();
    }

    public long getTimeDelta() {
        return System.currentTimeMillis() - time_;
    }
}
