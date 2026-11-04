//! Where training runs. Milestone 1 trains in this browser's Web Worker;
//! the remote training servers of Milestone 7.5 implement the same trait,
//! so the UI does not change when a server is picked.

use std::cell::Cell;
use std::rc::Rc;

use crate::protocol::SimRequest;

/// A training job split into chunks: the UI stays responsive, progress
/// arrives after every chunk, and the job can stop between chunks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrainingJob {
    remaining: u32,
}

impl TrainingJob {
    pub fn new(episodes: u32) -> Self {
        Self { remaining: episodes }
    }

    /// The next chunk of at most `chunk` episodes, or `None` when done.
    pub fn next_chunk(&mut self, chunk: u32) -> Option<u32> {
        let n = self.remaining.min(chunk);
        self.remaining -= n;
        (n > 0).then_some(n)
    }

    pub fn cancel(&mut self) {
        self.remaining = 0;
    }
}

pub trait TrainingTarget {
    /// Shown in the status bar, e.g. "this browser · Web Worker".
    fn label(&self) -> String;
    /// Trains the current raft for `episodes` more episodes.
    fn start(&self, episodes: u32);
    /// Stops once the chunk in flight returns.
    fn cancel(&self);
    /// Called when a chunk's results arrive. Returns `true` when the job is over.
    fn chunk_done(&self) -> bool;
}

/// Training in the simulation Web Worker on this machine.
pub struct LocalWorker {
    job: Cell<TrainingJob>,
    chunk: u32,
    send: Rc<dyn Fn(SimRequest)>,
}

impl LocalWorker {
    pub fn new(chunk: u32, send: Rc<dyn Fn(SimRequest)>) -> Self {
        Self { job: Cell::new(TrainingJob::default()), chunk, send }
    }

    fn send_next(&self) -> bool {
        let mut job = self.job.get();
        let next = job.next_chunk(self.chunk);
        self.job.set(job);
        match next {
            Some(episodes) => {
                (self.send)(SimRequest::Train { episodes });
                true
            }
            None => false,
        }
    }
}

impl TrainingTarget for LocalWorker {
    fn label(&self) -> String {
        "this browser · Web Worker".into()
    }

    fn start(&self, episodes: u32) {
        self.job.set(TrainingJob::new(episodes));
        self.send_next();
    }

    fn cancel(&self) {
        self.job.set(TrainingJob::default());
    }

    fn chunk_done(&self) -> bool {
        !self.send_next()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    #[test]
    fn jobs_run_in_chunks_and_stop_on_cancel() {
        let sent = Rc::new(RefCell::new(Vec::new()));
        let log = sent.clone();
        let target = LocalWorker::new(50, Rc::new(move |r| log.borrow_mut().push(r)));
        target.start(120);
        assert!(!target.chunk_done());
        assert!(!target.chunk_done());
        assert!(target.chunk_done(), "120 episodes take three chunks");
        let episodes: Vec<u32> = sent
            .borrow()
            .iter()
            .map(|r| match r {
                SimRequest::Train { episodes } => *episodes,
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert_eq!(episodes, vec![50, 50, 20]);
        target.start(500);
        target.cancel();
        assert!(target.chunk_done());
        assert_eq!(sent.borrow().len(), 4);
    }
}
