
use std::sync::atomic::{AtomicBool, Ordering};

pub struct RiskEngine {
    pub kill_switch: AtomicBool,
    pub max_notional: u64,
}

impl RiskEngine {
    pub fn new(max_notional: u64) -> Self {
        Self {
            kill_switch: AtomicBool::new(false),
            max_notional,
        }
    }

    pub fn check_order(&self, value: u64) -> bool {
        if self.kill_switch.load(Ordering::SeqCst) {
            return false;
        }
        if value > self.max_notional {
            return false;
        }
        true
    }

    pub fn trigger_kill_switch(&self) {
        self.kill_switch.store(true, Ordering::SeqCst);
        // In real hardware, this would trip a relay or send a TCP RST
    }
}
