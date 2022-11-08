mod map;
mod stack;

pub use map::ConcurrentMap;
pub use stack::{ConcurrentStack, ConcurrentStackPusher};

/*
fn dynamic_is_copy<T: ?Sized>() -> bool {
    struct Sentinel<'a, T: ?Sized> {
        true_if_copy: &'a std::cell::Cell<bool>,
        _marker: std::marker::PhantomData<T>,
    }

    impl<T: ?Sized> Clone for Sentinel<'_, T> {
        fn clone(&self) -> Self {
            self.true_if_copy.set(false);
            Sentinel {
                true_if_copy: self.true_if_copy,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<T: ?Sized + Copy> Copy for Sentinel<'_, T> {}

    let true_if_copy = std::cell::Cell::new(true);

    // array implements a Clone specialization where
    // it does a plain copy instead of Clone if the
    // inner type implements Copy. This means that the
    // actual Clone implementation is skipped.
    let _ = [Sentinel::<T> {
        true_if_copy: &true_if_copy,
        _marker: std::marker::PhantomData,
    }]
    .clone();

    true_if_copy.get()
}

#[test]
fn dyno() {
    dbg!(dynamic_is_copy::<usize>());
    dbg!(dynamic_is_copy::<String>());
}
*/
