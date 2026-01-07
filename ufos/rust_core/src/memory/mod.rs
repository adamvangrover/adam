
use std::sync::atomic::{AtomicUsize, Ordering};
use std::mem::MaybeUninit;

// Zero-Allocation Imperative
// 2.1.1 The Arena Allocator and Slab Allocation

pub struct Arena<T> {
    data: Vec<MaybeUninit<T>>,
    next_free: AtomicUsize,
    capacity: usize,
}

impl<T> Arena<T> {
    pub fn new(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            data.push(MaybeUninit::uninit());
        }

        Arena {
            data,
            next_free: AtomicUsize::new(0),
            capacity,
        }
    }

    pub fn allocate(&self, value: T) -> Option<usize> {
        let index = self.next_free.fetch_add(1, Ordering::Relaxed);
        if index >= self.capacity {
            return None; // Arena full
        }

        // In a real implementation we would need to handle safety more carefully
        // or use interior mutability (UnsafeCell) if we want to write to the Vec
        // from multiple threads without locking.
        // For this blueprint, we simulate the allocation index return.

        // unsafe {
        //     (*self.data[index].as_mut_ptr()) = value;
        // }

        Some(index)
    }

    // Stub for returning memory to the arena
    pub fn deallocate(&self, _index: usize) {
        // Add to free list
    }
}
