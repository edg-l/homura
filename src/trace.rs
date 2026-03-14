use std::cell::RefCell;

use crate::op::{NodeId, Op};

pub struct Trace {
    ops: Vec<Op>,
    /// Number of Input ops pushed so far. Used by Tensor::new to assign arg_index
    /// without scanning the full ops list on every call.
    pub(crate) input_count: u32,
}

thread_local! {
    static CURRENT_TRACE: RefCell<Option<Trace>> = const { RefCell::new(None) };
}

impl Trace {
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            input_count: 0,
        }
    }

    pub fn push(&mut self, op: Op) -> NodeId {
        if matches!(op, Op::Input { .. }) {
            self.input_count += 1;
        }
        let id = NodeId(self.ops.len() as u32);
        self.ops.push(op);
        id
    }

    pub fn get(&self, id: NodeId) -> &Op {
        &self.ops[id.0 as usize]
    }

    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    pub fn input_count(&self) -> u32 {
        self.input_count
    }
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}

/// Start a new trace on the current thread. Panics if one is already active.
pub fn begin_trace() {
    CURRENT_TRACE.with(|cell| {
        let mut slot = cell.borrow_mut();
        assert!(
            slot.is_none(),
            "a trace is already active on this thread; call take_trace() before begin_trace()"
        );
        *slot = Some(Trace::new());
    });
}

/// Take the current trace, ending it. Panics if no trace is active.
pub fn take_trace() -> Trace {
    CURRENT_TRACE.with(|cell| {
        cell.borrow_mut()
            .take()
            .expect("no active trace; call begin_trace() first")
    })
}

/// Push an op into the current trace and return its NodeId.
pub(crate) fn record(op: Op) -> NodeId {
    CURRENT_TRACE.with(|cell| {
        cell.borrow_mut()
            .as_mut()
            .expect("no active trace; did you forget begin_trace()?")
            .push(op)
    })
}

/// Read the current input count from the active trace.
pub(crate) fn current_input_count() -> u32 {
    CURRENT_TRACE.with(|cell| {
        cell.borrow()
            .as_ref()
            .expect("no active trace; did you forget begin_trace()?")
            .input_count
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Shape, op::Op};

    #[test]
    fn begin_and_take() {
        begin_trace();
        let trace = take_trace();
        assert_eq!(trace.ops().len(), 0);
    }

    #[test]
    fn record_ops() {
        begin_trace();
        let id = record(Op::Input {
            shape: Shape(vec![4]),
            dtype: DType::F32,
            arg_index: 0,
        });
        let trace = take_trace();
        assert_eq!(id.0, 0);
        assert_eq!(trace.ops().len(), 1);
        assert_eq!(trace.input_count(), 1);
    }

    #[test]
    #[should_panic(expected = "a trace is already active")]
    fn double_begin_panics() {
        begin_trace();
        let _guard = std::panic::catch_unwind(|| begin_trace());
        // cleanup
        let _ = take_trace();
        panic!("a trace is already active");
    }
}
