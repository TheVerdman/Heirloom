use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

pub fn no_grad<R>(f: impl FnOnce() -> R) -> R {
    let previous = GRAD_ENABLED.with(|enabled| {
        let previous = enabled.get();
        enabled.set(false);
        previous
    });

    let result = f();
    GRAD_ENABLED.with(|enabled| enabled.set(previous));
    result
}

pub(crate) fn should_track_grad(requires_grad: bool) -> bool {
    requires_grad && is_grad_enabled()
}
