use std::mem::ManuallyDrop;
use std::sync::{
    atomic::{AtomicPtr, Ordering},
    Arc,
};

use ebr::Ebr;

#[derive(Debug)]
struct Node<T: Send> {
    next: *mut Node<T>,
    item: ManuallyDrop<T>,
}

unsafe impl<T: Send> Send for Node<T> {}

#[derive(Default, Clone, Debug)]
pub struct Pusher<T: 'static + Send> {
    head: Arc<AtomicPtr<Node<T>>>,
}

impl<T: Send> Drop for Pusher<T> {
    fn drop(&mut self) {
        if Arc::strong_count(&self.head) != 1 {
            return;
        }

        let mut cursor = self.head.load(Ordering::Acquire);
        while !cursor.is_null() {
            let mut node: Box<Node<T>> = unsafe { Box::from_raw(cursor) };
            unsafe {
                ManuallyDrop::drop(&mut node.item);
            }
            cursor = node.next;
        }
    }
}

impl<T: Send> Pusher<T> {
    pub fn push(&self, item: T) {
        let mut head = self.head.load(Ordering::Acquire);

        let node = Box::new(Node {
            item: ManuallyDrop::new(item),
            next: head,
        });

        let node_ptr = Box::into_raw(node);

        loop {
            let install_res =
                self.head
                    .compare_exchange(head, node_ptr, Ordering::AcqRel, Ordering::Acquire);

            match install_res {
                Ok(_) => return,
                Err(actual_head) => {
                    head = actual_head;

                    unsafe {
                        (*node_ptr).next = head;
                    }
                }
            }
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct Stack<T: 'static + Send> {
    head: Arc<AtomicPtr<Node<T>>>,
    ebr: Ebr<Box<Node<T>>>,
}

impl<T: Send> Drop for Stack<T> {
    fn drop(&mut self) {
        if Arc::strong_count(&self.head) != 1 {
            return;
        }

        let mut cursor = self.head.load(Ordering::Acquire);
        while !cursor.is_null() {
            let mut node: Box<Node<T>> = unsafe { Box::from_raw(cursor) };
            unsafe {
                ManuallyDrop::drop(&mut node.item);
            }
            cursor = node.next;
        }
    }
}

impl<T: Send> Stack<T> {
    pub fn push(&self, item: T) {
        let mut head = self.head.load(Ordering::Acquire);

        let node = Box::new(Node {
            item: ManuallyDrop::new(item),
            next: head,
        });

        let node_ptr = Box::into_raw(node);

        loop {
            let install_res =
                self.head
                    .compare_exchange(head, node_ptr, Ordering::AcqRel, Ordering::Acquire);

            match install_res {
                Ok(_) => return,
                Err(actual_head) => {
                    head = actual_head;

                    unsafe {
                        (*node_ptr).next = head;
                    }
                }
            }
        }
    }

    pub fn pop(&self) -> Option<T> {
        let mut guard = self.ebr.pin();

        let mut cursor = self.head.load(Ordering::Acquire);
        while !cursor.is_null() {
            let head_node = unsafe { &*cursor };
            let head_next = head_node.next;

            let pop_res =
                self.head
                    .compare_exchange(cursor, head_next, Ordering::AcqRel, Ordering::Acquire);

            match pop_res {
                Ok(_) => {
                    let mut head: Box<Node<T>> = unsafe { Box::from_raw(cursor) };
                    let ret = unsafe { ManuallyDrop::take(&mut head.item) };
                    guard.defer_drop(head);
                    return Some(ret);
                }
                Err(actual_head) => {
                    cursor = actual_head;
                }
            }
        }

        None
    }

    /// Returns a push-only stack handle. This is better
    /// to use where possible, because it avoids registering
    /// shared state with the epoch-based reclamation system.
    pub fn get_pusher(&self) -> Pusher<T> {
        Pusher {
            head: self.head.clone(),
        }
    }
}

#[test]
fn basic_stack() {
    const N: usize = 128;

    let stack = Stack::<String>::default();
    for _ in 0..N {
        stack.push("yo".into());
    }
    for _ in 0..N {
        assert_eq!(stack.pop().unwrap(), "yo");
    }

    assert_eq!(stack.pop(), None);
}
